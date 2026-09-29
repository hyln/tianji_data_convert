// Offline FK only; no ROS node, driver, communication or robot command interface.
#include "FxRobot.h"
#include <cmath>

extern "C" int metrics_init(char *config) {
    FX_LOG_SWITCH(false);
    FX_INT32L type[2] = {};
    FX_DOUBLE gravity[2][3] = {}, dh[2][8][4] = {}, pnva[2][7][4] = {};
    FX_DOUBLE bd[2][4][3] = {}, mass[2][7] = {}, mcp[2][7][3] = {}, inertia[2][7][6] = {};
    if (!LOADMvCfg(config, type, gravity, dh, pnva, bd, mass, mcp, inertia)) return 0;
    for (int arm = 0; arm < 2; ++arm) {
        if (!FX_Robot_Init_Type(arm, type[arm]) || !FX_Robot_Init_Kine(arm, dh[arm])) return 0;
    }
    return 1;
}

extern "C" int metrics_fk(int arm, int n, const double *radians, double *xyz_mm) {
    constexpr double rad_to_deg = 180.0 / 3.14159265358979323846;
    for (int row = 0; row < n; ++row) {
        double joints[7];
        for (int j = 0; j < 7; ++j) joints[j] = radians[row * 7 + j] * rad_to_deg;
        Matrix4 tcp{};
        if (!FX_Robot_Kine_FK(arm, joints, tcp)) return 0;
        // Exact position conversion from motion_control/src/ik.cpp::ikMatrixToPose.
        xyz_mm[row * 3] = tcp[0][3];
        xyz_mm[row * 3 + 1] = arm == 1 ? -tcp[2][3] : tcp[2][3];
        xyz_mm[row * 3 + 2] = (arm == 1 ? tcp[1][3] : -tcp[1][3]) + 925.5;
    }
    return 1;
}
