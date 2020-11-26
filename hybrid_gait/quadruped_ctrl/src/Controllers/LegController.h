/*
 * @Author: your name
 * @Date: 2020-09-25 15:34:19
 * @LastEditTime: 2020-10-24 14:43:33
 * @LastEditors: Please set LastEditors
 * @Description: In User Settings Edit
 * @FilePath: /quadruped_ctrl/src/Controllers/LegController.h
 */
/*! @file LegController.h
 *  @brief Common Leg Control Interface and Leg Control Algorithms
 *
 *  Implements low-level leg control for Mini Cheetah and Cheetah 3 Robots
 *  Abstracts away the difference between the SPIne and the TI Boards (the low
 * level leg control boards) All quantities are in the "leg frame" which has the
 * same orientation as the body frame, but is shifted so that 0,0,0 is at the
 * ab/ad pivot (the "hip frame").
 */

#ifndef PROJECT_LEGCONTROLLER_H
#define PROJECT_LEGCONTROLLER_H

#include "Dynamics/Quadruped.h"
#include "RobotLegState.h"
#include "Utilities/cppTypes.h"
// #include "SimUtilities/SpineBoard.h"
// #include "SimUtilities/ti_boardcontrol.h"

/*!
 * Data sent from the control algorithm to the legs.
 * 控制算法计算出来的足底位置、速度、力矩值发送给腿部
 */
template <typename T>
struct LegControllerCommand {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  LegControllerCommand() { zero(); }

  void zero();

  Vec3<T> tauFeedForward, forceFeedForward, qDes, qdDes, pDes, vDes;
  Mat3<T> kpCartesian, kdCartesian, kpJoint, kdJoint;
};

/*!
 * Data returned from the legs to the control code.
 * 腿部反馈给控制代码的关节实时数据
 */
template <typename T>
struct LegControllerData {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  LegControllerData() { zero(); }

  void setQuadruped(Quadruped<T>& quad) { quadruped = &quad; }

  void zero();

  Vec3<T> q, qd, p, v;
  Mat3<T> J;
  Vec3<T> tauEstimate;
  Quadruped<T>* quadruped;
};

/*!
 * Controller for 4 legs of a quadruped.  Works for both Mini Cheetah and
 * Cheetah 3
 */
template <typename T>
class LegController {
 public:
  LegController(Quadruped<T>& quad) : _quadruped(quad) {
    for (auto& data : datas) data.setQuadruped(_quadruped);
  }

  void zeroCommand();
  void edampCommand(RobotType robot, T gain);
  void updateData(LegData* legData);
  // void updateData(const TiBoardData* tiBoardData);
  void updateCommand(LegCommand* legCommand, Vec4<T>& crtlParam);
  // void updateCommand(TiBoardCommand* tiBoardCommand);
  void setEnabled(bool enabled) { _legsEnabled = enabled; };
  // void setLcm(leg_control_data_lcmt* data, leg_control_command_lcmt*
  // command);

  /*!
   * Set the maximum torque.  This only works on cheetah 3!
   */
  void setMaxTorqueCheetah3(T tau) { _maxTorque = tau; }

  LegControllerCommand<T> commands[4];
  LegControllerData<T> datas[4];
  Quadruped<T>& _quadruped;
  bool _legsEnabled = false;
  T _maxTorque = 0;
  bool _zeroEncoders = false;
  u32 _calibrateEncoders = 0;
  int flags = 0;

  float stand_target[12] = {0.0, -0.8, 1.6, 0.0, -0.8, 1.6,
                            0.0, -0.8, 1.6, 0.0, -0.8, 1.6};
  float init_pos[12] = {0.0};

  int myflags = 0;
};

template <typename T>
void computeLegJacobianAndPosition(Quadruped<T>& quad, Vec3<T>& q, Mat3<T>* J,
                                   Vec3<T>* p, int leg);

#endif  // PROJECT_LEGCONTROLLER_H
