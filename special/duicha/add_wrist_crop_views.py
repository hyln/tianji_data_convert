#!/usr/bin/env python3
"""Incrementally add wrist crop videos to an existing three-view LeRobot v3 dataset."""
from __future__ import annotations
import argparse,copy,csv,hashlib,json,shutil,time
from concurrent.futures import ThreadPoolExecutor,as_completed
from fractions import Fraction
from pathlib import Path
import sys

# Direct CLI execution must be able to import the shared conversion modules.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import av,cv2,numpy as np,pyarrow as pa,pyarrow.parquet as pq
from common.camera_metadata import load_camera_intrinsics
from common.wrist_undistortion import DirectCropper
from common.feedback_state import file_sha256,source_fingerprint

REVISION='incremental_wrist_crop_v13'
VIEWS=('hik_camera_6_crop','hik_camera_7_crop')
BASE_VIEWS=('head','hik_camera_6','hik_camera_7')

def json_write(path,data):
 path.parent.mkdir(parents=True,exist_ok=True)
 temp=path.with_name(path.name+'.tmp');temp.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n');temp.replace(path)

def histogram_stats(hist):
 """Exact RGB pixel statistics; float arithmetic avoids uint8 square overflow."""
 hist=np.asarray(hist,dtype=np.int64);totals=hist.sum(axis=1)
 if hist.shape!=(3,256) or not np.all(totals==totals[0]) or totals[0]<=0:raise ValueError('invalid RGB histogram')
 n=int(totals[0]);levels=np.arange(256,dtype=np.float64)/255
 mean=(hist*levels).sum(axis=1)/n;variance=(hist*(levels[None,:]-mean[:,None])**2).sum(axis=1)/n
 values={'min':np.array([np.flatnonzero(h)[0] for h in hist])/255,'max':np.array([np.flatnonzero(h)[-1] for h in hist])/255,'mean':mean,'std':np.sqrt(variance)}
 cumulative=hist.cumsum(axis=1)
 for q in (.01,.10,.50,.90,.99):
  rank=(n-1)*q;lo=int(np.floor(rank));hi=int(np.ceil(rank));fraction=rank-lo
  lower=np.array([np.searchsorted(c,lo,side='right') for c in cumulative]);upper=np.array([np.searchsorted(c,hi,side='right') for c in cumulative]);values[f'q{int(q*100):02d}']=(lower*(1-fraction)+upper*fraction)/255
 return {'count':[n],**{k:v.reshape(3,1,1).tolist() for k,v in values.items()}}

def read_timestamps(path):
 with path.open() as f:rows=list(csv.DictReader(f))
 if [int(r['frame_index']) for r in rows]!=list(range(len(rows))):raise ValueError('non-contiguous raw frame index')
 times=np.array([int(r['header_stamp_ns'] or r['receive_timestamp_ns']) for r in rows],np.int64)
 if len(times)==0 or np.any(np.diff(times)<0):raise ValueError('invalid raw camera timestamps')
 return times

def probe_video(path):
 with av.open(str(path)) as container:
  stream=container.streams.video[0]
  return {'frames':stream.frames,'width':stream.width,'height':stream.height,'codec':stream.codec_context.name}

def encode_crop(source,camera,stamps,destination,calibration,fps,crf):
 stem=f'perception__sensors__hik_camera_{camera}__image';times=read_timestamps(source/'topic_csv'/(stem+'__frames.csv'))
 wanted=np.searchsorted(times,stamps,side='right')-1
 if np.any(wanted<0) or np.any(np.diff(wanted)<0):raise ValueError('crop cannot align to existing dataset timestamps')
 processor=DirectCropper(calibration,f'hik_camera_{camera}',224);hist=np.zeros((3,256),np.int64)
 cap=cv2.VideoCapture(str(source/'video'/(stem+'.mp4')))
 if not cap.isOpened():raise RuntimeError('cannot open raw wrist video')
 destination.parent.mkdir(parents=True,exist_ok=True);temporary=destination.with_name(destination.stem+'.partial.mp4')
 try:
  with av.open(str(temporary),'w') as container:
   stream=container.add_stream('libx264',fps,options={'crf':str(crf),'g':'2','threads':'1'});stream.pix_fmt='yuv420p';stream.width=224;stream.height=224;stream.time_base=Fraction(1,fps)
   decoded=-1;bgr=None;last=-1;rgb=None
   for i,target in enumerate(wanted):
    while decoded<int(target):
     ok,bgr=cap.read();decoded+=1
     if not ok:raise RuntimeError(f'raw video ended at {decoded}, expected {target}')
    if int(target)!=last:rgb=cv2.cvtColor(processor(bgr),cv2.COLOR_BGR2RGB);last=int(target)
    video_frame=av.VideoFrame.from_ndarray(rgb,format='rgb24');video_frame.pts=i;video_frame.time_base=Fraction(1,fps)
    for packet in stream.encode(video_frame):container.mux(packet)
    for channel in range(3):hist[channel]+=np.bincount(rgb[:,:,channel].reshape(-1),minlength=256)
   for packet in stream.encode():container.mux(packet)
  meta=probe_video(temporary)
  if meta!={'frames':len(stamps),'width':224,'height':224,'codec':'h264'}:raise RuntimeError(f'encoded video mismatch: {meta}')
  temporary.replace(destination)
 finally:cap.release()
 return hist,{'first_raw_frame':int(wanted[0]),'last_raw_frame':int(wanted[-1]),'aligned_frame_index_sha256':hashlib.sha256(wanted.astype('<i8').tobytes()).hexdigest()}

def validate_reusable_base(base,calibration):
 info=json.loads((base/'meta/info.json').read_text());old=json.loads((base/'meta/camera_intrinsics.json').read_text())
 expected={'observation.images.'+name for name in BASE_VIEWS}
 if {k for k in info['features'] if k.startswith('observation.images.')}!=expected:raise ValueError('base must have exactly the established three views')
 if old.get('image_preprocessing',{}).get('mode') is not None:raise ValueError('base must be the full-view resize dataset')
 if old['source_camera_matrix']!=calibration['source_camera_matrix']:raise ValueError('base source K differs')
 if (old['source_image_width'],old['source_image_height'])!=(calibration['source_image_width'],calibration['source_image_height']):raise ValueError('base source resolution differs')
 if old.get('undistortion_applied')!={'observation.images.head':False,'observation.images.hik_camera_6':True,'observation.images.hik_camera_7':True}:raise ValueError('base must have corrected wrists and original head')
 for key in ('model','coefficients','supersampling_factor'):
  if old['wrist_undistortion'][key]!=calibration['wrist_undistortion'][key]:raise ValueError('base wrist processing differs')
 for name in BASE_VIEWS:
  if info['features']['observation.images.'+name]['shape']!=[224,224,3]:raise ValueError('base image size differs')
  if old['camera_matrix']!=calibration['per_camera_intrinsics']['observation.images.'+name]['camera_matrix']:raise ValueError('base output K differs')
 return info

def augment(base,output,calibration_path,workers=8,crf=18):
 started=time.monotonic();base=base.resolve();output=output.resolve();calibration_path=calibration_path.resolve()
 if base==output or base in output.parents or output in base.parents:raise ValueError('input and output directories must be separate')
 if output.exists():raise FileExistsError(output)
 calibration=load_camera_intrinsics(calibration_path,224)
 if calibration['image_preprocessing']['mode']!='five_view':raise ValueError('five-view calibration required')
 info=validate_reusable_base(base,calibration);fps=int(info['fps']);chunks=int(info['chunks_size'])
 partial=output.with_name(output.name+'.incremental.partial');partial.mkdir(parents=True,exist_ok=True)
 signature={'revision':REVISION,'base':str(base),'base_info_sha256':file_sha256(base/'meta/info.json'),'base_annotations_sha256':file_sha256(base/'meta/source_annotations.json'),'camera_config_sha256':file_sha256(calibration_path),'crf':crf}
 signature_path=partial/'incremental_signature.json'
 if signature_path.exists() and json.loads(signature_path.read_text())!=signature:raise RuntimeError('incremental partial has different input or recipe')
 json_write(signature_path,signature)
 copied=[]
 # ExFAT has no hardlinks. Copies preserve existing video bytes; no decoding or encoding.
 for p in sorted(base.rglob('*')):
  if not p.is_file():continue
  relative=p.relative_to(base)
  if relative.parts[0] not in ('data','videos','meta'):continue
  target=partial/relative;target.parent.mkdir(parents=True,exist_ok=True)
  digest=file_sha256(p)
  if not target.exists() or file_sha256(target)!=digest:shutil.copy2(p,target)
  if file_sha256(target)!=digest:raise RuntimeError(f'copy mismatch: {relative}')
  copied.append({'path':str(relative),'sha256':digest,'bytes':p.stat().st_size})
 print(f'[reuse] {base.name}: copied and hashed {len(copied)} existing files',flush=True)
 source_meta=json.loads((base/'meta/source_annotations.json').read_text());table=pa.concat_tables([pq.read_table(p) for p in sorted((base/'data').rglob('*.parquet'))]);episodes=np.array(table['episode_index'].to_pylist());stamps=np.array(table['source.timestamp_ns'].to_pylist(),np.int64).reshape(-1)
 if len(table)!=info['total_frames'] or len(source_meta['sources'])!=info['total_episodes']:raise ValueError('base counts mismatch')
 job_root=partial/'incremental_jobs';job_root.mkdir(exist_ok=True)
 def job(source):
  index=int(source['episode_index']);raw=Path(source['source_directory']);selected=stamps[episodes==index];report_path=job_root/f'{index:06d}.json'
  if not len(selected):raise ValueError('base episode has no frames')
  raw_fingerprint=source_fingerprint(raw);stamp_hash=hashlib.sha256(selected.astype('<i8').tobytes()).hexdigest()
  if raw_fingerprint!=source['worker_report'].get('source_fingerprint'):raise RuntimeError('raw source changed since base conversion: '+source['source_id'])
  expected={'signature':signature,'source_id':source['source_id'],'source_fingerprint':raw_fingerprint,'timestamp_sha256':stamp_hash,'frames':len(selected),'episode_index':index}
  if report_path.exists():
   report=json.loads(report_path.read_text())
   if all(report.get(k)==v for k,v in expected.items()) and all((partial/v['path']).exists() and file_sha256(partial/v['path'])==v['sha256'] for v in report['videos'].values()):return report
  report={**expected,'videos':{}}
  for camera in (6,7):
   name=f'hik_camera_{camera}_crop';key='observation.images.'+name;relative=Path(info['video_path'].format(video_key=key,chunk_index=index//chunks,file_index=index%chunks))
   hist,alignment=encode_crop(raw,camera,selected,partial/relative,calibration,fps,crf)
   report['videos'][key]={'path':str(relative),'sha256':file_sha256(partial/relative),'histogram':hist.tolist(),'stats':histogram_stats(hist),'alignment':alignment,'crop_xywh':calibration['image_preprocessing']['crop_rectangles'][f'hik_camera_{camera}']}
  json_write(report_path,report);return report
 results=[]
 with ThreadPoolExecutor(max_workers=workers) as pool:
  futures=[pool.submit(job,s) for s in source_meta['sources']]
  for future in as_completed(futures):
   report=future.result();results.append(report);print(f"[augment {len(results)}/{info['total_episodes']}] {report['source_id']} frames={report['frames']}",flush=True)
 results.sort(key=lambda r:r['episode_index']);by_index={r['episode_index']:r for r in results}
 for p in sorted((base/'meta/episodes').rglob('*.parquet')):
  original=pq.read_table(p);indices=original['episode_index'].to_pylist();updated=original
  for name in VIEWS:
   key='observation.images.'+name
   fields={'chunk_index':([i//chunks for i in indices],pa.int64()),'file_index':([i%chunks for i in indices],pa.int64()),'from_timestamp':([0.0]*len(indices),pa.float64()),'to_timestamp':([by_index[i]['frames']/fps for i in indices],pa.float64())}
   for field,(values,dtype) in fields.items():updated=updated.append_column(f'videos/{key}/{field}',pa.array(values,type=dtype))
   for field in by_index[indices[0]]['videos'][key]['stats']:
    values=[by_index[i]['videos'][key]['stats'][field] for i in indices];updated=updated.append_column(f'stats/{key}/{field}',pa.array(values))
  if not updated.select(original.column_names).equals(original):raise RuntimeError('original episode metadata values changed')
  pq.write_table(updated.replace_schema_metadata(None),partial/p.relative_to(base),compression='snappy')
 stats=json.loads((base/'meta/stats.json').read_text())
 for name in VIEWS:
  key='observation.images.'+name;hist=np.sum([np.array(r['videos'][key]['histogram'],np.int64) for r in results],axis=0);stats[key]=histogram_stats(hist)
  info['features'][key]=copy.deepcopy(info['features']['observation.images.'+name.removesuffix('_crop')])
 json_write(partial/'meta/stats.json',stats);json_write(partial/'meta/info.json',info);json_write(partial/'meta/camera_intrinsics.json',calibration)
 for source in source_meta['sources']:
  r=by_index[source['episode_index']];source['crop_augmentation']={k:v for k,v in r.items() if k!='videos'};source['crop_augmentation']['videos']={k:{f:v for f,v in data.items() if f not in ('histogram','stats')} for k,data in r['videos'].items()}
 source_meta['incremental_augmentation']={'revision':REVISION,'base_dataset':str(base),'added_views':list(VIEWS),'reencoded_existing_views':False}
 json_write(partial/'meta/source_annotations.json',source_meta)
 old_report=json.loads((base/'conversion_report.json').read_text());json_write(partial/'meta/base_conversion_report.json',old_report)
 report={**old_report,'conversion_revision':REVISION,'incremental':True,'base_dataset':str(base),'output':str(output),'elapsed_s':time.monotonic()-started,'added_views':list(VIEWS),'reencoded_existing_views':False,'recomputed_state':False,'statistics_method':'exact per-channel 256-bin uint8 histogram of every crop pixel before H264 encoding; float64 moments; count counts pixels','reused_files':copied,'camera_intrinsics_sha256':file_sha256(calibration_path),'total_episodes':info['total_episodes'],'total_frames':info['total_frames']}
 # Prove every original video and data file remains byte-identical before publishing.
 for item in copied:
  if Path(item['path']).parts[0] in ('data','videos') and file_sha256(partial/item['path'])!=item['sha256']:raise RuntimeError('reused file changed')
 json_write(partial/'conversion_report.json',report);partial.rename(output)
 print(f"[complete] {output.name} episodes={info['total_episodes']} frames={info['total_frames']}",flush=True)
 return report

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--base',type=Path);parser.add_argument('--output',type=Path);parser.add_argument('--camera-intrinsics',type=Path);parser.add_argument('--workers',type=int,default=8);parser.add_argument('--config',type=Path);args=parser.parse_args();cv2.setNumThreads(1)
 if args.config:
  import yaml
  config=yaml.safe_load(args.config.read_text());cal=Path(config['camera_intrinsics']);cal=cal if cal.is_absolute() else args.config.resolve().parent/cal
  for job in config['datasets']:augment(Path(config['base_root'])/job['base'],Path(config['output_root'])/job['output'],cal,int(config.get('workers',8)),int(config.get('video_crf',18)))
 else:
  if not all((args.base,args.output,args.camera_intrinsics)):parser.error('--base, --output, --camera-intrinsics required')
  augment(args.base,args.output,args.camera_intrinsics,args.workers)
if __name__=='__main__':main()
