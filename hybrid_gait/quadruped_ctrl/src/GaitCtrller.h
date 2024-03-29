#ifndef GAIT_CTRLLER_H
#define GAIT_CTRLLER_H

#include <math.h>
#include <time.h>

#include <iostream>
#include <string>

#include "Controllers/ContactEstimator.h"
#include "Controllers/ControlFSMData.h"
#include "Controllers/DesiredStateCommand.h"
#include "Controllers/OrientationEstimator.h"
#include "Controllers/PositionVelocityEstimator.h"
#include "Controllers/RobotLegState.h"
#include "Controllers/StateEstimatorContainer.h"
#include "Controllers/SafetyChecker.h"
#include "Dynamics/MiniCheetah.h"
#include "MPC_Ctrl/ConvexMPCLocomotion.h"
#include "Utilities/IMUTypes.h"
#include "calculateTool.h"

struct JointEff {
  double eff[12];
};

class GaitCtrller {
 public:
  GaitCtrller(double freq, double* PIDParam);
  ~GaitCtrller();
  void SetIMUData(double* imuData);
  void SetLegData(double* motorData);
  void PreWork(double* imuData, double* motorData);
  int GetSafetyCheck();
  double GetPrfFootCoor();
  void SetGaitParam(int* gaitType);
  void SetGaitType(int gaitType);
  void SetRobotMode(int mode);
  void SetRobotVel(double* vel);
  void ToqueCalculator(double* imuData, double* motorData, double* effort);

 private:
  int _gaitType = 0;
  int _robotMode = 2;
  bool _safetyCheck = true;
  double _footXCoor = 0.0;
  std::vector<double> _gamepadCommand;
  std::vector<int> _gaitParam;
  Vec4<float> ctrlParam;

  Quadruped<float> _quadruped;
  ConvexMPCLocomotion* convexMPC;
  LegController<float>* _legController;
  StateEstimatorContainer<float>* _stateEstimator;
  LegData _legdata;
  LegCommand legcommand;
  ControlFSMData<float> control_data;
  VectorNavData _vectorNavData;
  CheaterState<double>* cheaterState;
  StateEstimate<float> _stateEstimate;
  RobotControlParameters* controlParameters;
  DesiredStateCommand<float>* _desiredStateCommand;
  SafetyChecker<float>* safetyChecker;
};

extern "C" {

GaitCtrller* gCtrller = NULL;
JointEff jointEff;

// first step, init the controller
void init_controller(double freq, double PIDParam[]) {
  if (NULL != gCtrller) {
    delete gCtrller;
  }
  gCtrller = new GaitCtrller(freq, PIDParam);
}

// the kalman filter need to work second
void pre_work(double imuData[], double legData[]) {
  gCtrller->PreWork(imuData, legData);
}

// gait params can be set in any time
int get_safety_check() { return gCtrller->GetSafetyCheck(); }

// get avg foot coor
double get_prf_foot_coor() { return gCtrller->GetPrfFootCoor(); }

// gait params can be set in any time
void set_gait_param(int gaitParam[]) { gCtrller->SetGaitParam(gaitParam); }

// gait type can be set in any time
void set_gait_type(int gaitType) { gCtrller->SetGaitType(gaitType); }

// set robot mode, 0: High performance model, 1: Low power mode
void set_robot_mode(int mode) { gCtrller->SetRobotMode(mode); }

// robot vel can be set in any time
void set_robot_vel(double vel[]) { gCtrller->SetRobotVel(vel); }

// after init controller and pre work, the mpc calculator can work
JointEff* toque_calculator(double imuData[], double motorData[]) {
  double eff[12] = {0.0};
  gCtrller->ToqueCalculator(imuData, motorData, eff);
  // std::cout << "eff = ";
  for (int i = 0; i < 12; i++) {
    jointEff.eff[i] = eff[i];
    // std::cout << p->eff[i] << " ";
  }
  // std::cout << std::endl;
  return &jointEff;
}
}

#endif