#!/usr/bin/env python3

import os
import time
import yaml
import threading
import random
import ctypes
import numpy as np
import pybullet as p
import pybullet_data

get_last_vel = [0] * 3
robot_height = 0.30
motor_id_list = [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14]
init_new_pos = [0.0, -0.8, 1.6, 0.0, -0.8, 1.6, 0.0, -0.8, 1.6, 0.0, -0.8, 1.6,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
fix_vel_test = True

class StructPointer(ctypes.Structure):
    _fields_ = [("eff", ctypes.c_double * 12)]


def convert_type(input):
    ctypes_map = {int: ctypes.c_int,
                  float: ctypes.c_double,
                  str: ctypes.c_char_p
                  }
    input_type = type(input)
    if input_type is list:
        length = len(input)
        if length == 0:
            print("convert type failed...input is "+input)
            return 0
        else:
            arr = (ctypes_map[type(input[0])] * length)()
            for i in range(length):
                arr[i] = bytes(
                    input[i], encoding="utf-8") if (type(input[0]) is str) else input[i]
            return arr
    else:
        if input_type in ctypes_map:
            return ctypes_map[input_type](bytes(input, encoding="utf-8") if type(input) is str else input)
        else:
            print("convert type failed...input is "+input)
            return 0


def get_energy_consumption(motor_tau, motor_vel):
    engergy_consumption = 0.0
    for i in range(12):
        engergy_consumption += np.abs(motor_tau[i] * motor_vel[i]) * (1.0/freq)
    # engergy_consumption = np.abs(np.dot(motor_tau, motor_vel)) * (1.0/freq)
    return engergy_consumption


def get_target_vel(t):
    nf = 5
    w_f = 0.8
    q0 = [0.0]
    q = [0.0] * 3
    qq = 0
    # if t == 0:
    #     a = [random.uniform(-1,1)  for _ in range(nf)]
    #     b = [random.uniform(-1,1)  for _ in range(nf)]
    a = [0.2886,-0.2428,0.6232,0.0657,-0.2985]
    b = [0.8780,0.7519,0.1003,0.2450,0.1741]
    for l in range(2, nf):
        qq += (a[l-2]/(w_f*l))*np.sin(w_f*l*t) - (b[l-2]/(w_f*l))*np.cos(w_f*l*t)
        q0 += (a[l-2]/(w_f*l))*np.sin(w_f*l*0) - (b[l-2]/(w_f*l))*np.cos(w_f*l*0)
    qq -= q0
    q[0] = qq.tolist()[0]/2

    a = [-0.6342,-0.5201,0.7730,-0.9427,-0.0202]
    b = [-0.6641,0.9574,0.4254,0.0009,-0.0578]
    q0 = [0.0]
    qq = 0
    for l in range(2, nf):
        qq += (a[l-2]/(w_f*l))*np.sin(w_f*l*t) - (b[l-2]/(w_f*l))*np.cos(w_f*l*t)
        q0 += (a[l-2]/(w_f*l))*np.sin(w_f*l*0) - (b[l-2]/(w_f*l))*np.cos(w_f*l*0)
    qq -= q0
    q[1] = qq.tolist()[0]/2

    return q

def acc_filter(value, last_accValue):
    a = 1
    filter_value = a * value + (1 - a) * last_accValue
    return filter_value


def get_data_from_sim():
    global get_last_vel
    get_orientation = []
    get_matrix = []
    get_velocity = []
    get_invert = []
    imu_data = [0] * 10
    leg_data = [0] * 24

    pose_orn = p.getBasePositionAndOrientation(boxId)

    for i in range(4):
        get_orientation.append(pose_orn[1][i])
    # get_euler = p.getEulerFromQuaternion(get_orientation)
    get_velocity = p.getBaseVelocity(boxId)
    get_invert = p.invertTransform(pose_orn[0], pose_orn[1])
    get_matrix = p.getMatrixFromQuaternion(get_invert[1])

    # IMU data
    imu_data[3] = pose_orn[1][0]
    imu_data[4] = pose_orn[1][1]
    imu_data[5] = pose_orn[1][2]
    imu_data[6] = pose_orn[1][3]

    imu_data[7] = get_matrix[0] * get_velocity[1][0] + get_matrix[1] * \
        get_velocity[1][1] + get_matrix[2] * get_velocity[1][2]
    imu_data[8] = get_matrix[3] * get_velocity[1][0] + get_matrix[4] * \
        get_velocity[1][1] + get_matrix[5] * get_velocity[1][2]
    imu_data[9] = get_matrix[6] * get_velocity[1][0] + get_matrix[7] * \
        get_velocity[1][1] + get_matrix[8] * get_velocity[1][2]

    # calculate the acceleration of the robot
    linear_X = (get_velocity[0][0] - get_last_vel[0]) * freq
    linear_Y = (get_velocity[0][1] - get_last_vel[1]) * freq
    linear_Z = 9.8 + (get_velocity[0][2] - get_last_vel[2]) * freq
    imu_data[0] = get_matrix[0] * linear_X + \
        get_matrix[1] * linear_Y + get_matrix[2] * linear_Z
    imu_data[1] = get_matrix[3] * linear_X + \
        get_matrix[4] * linear_Y + get_matrix[5] * linear_Z
    imu_data[2] = get_matrix[6] * linear_X + \
        get_matrix[7] * linear_Y + get_matrix[8] * linear_Z

    # joint data
    joint_state = p.getJointStates(boxId, motor_id_list)
    leg_data[0:12] = [joint_state[0][0], joint_state[1][0], joint_state[2][0],
                      joint_state[3][0], joint_state[4][0], joint_state[5][0],
                      joint_state[6][0], joint_state[7][0], joint_state[8][0],
                      joint_state[9][0], joint_state[10][0], joint_state[11][0]]

    leg_data[12:24] = [joint_state[0][1], joint_state[1][1], joint_state[2][1],
                       joint_state[3][1], joint_state[4][1], joint_state[5][1],
                       joint_state[6][1], joint_state[7][1], joint_state[8][1],
                       joint_state[9][1], joint_state[10][1], joint_state[11][1]]
    com_velocity = [get_velocity[0][0],
                    get_velocity[0][1], get_velocity[0][2]]
    get_last_vel.clear()
    get_last_vel = com_velocity

    return imu_data, leg_data, pose_orn[0]


def reset_robot():
    p.resetBasePositionAndOrientation(
        boxId, [0, 0, robot_height], [0, 0, 0, 1])
    for j in range(12):
        p.resetJointState(boxId, motor_id_list[j], init_new_pos[j])
    cpp_gait_ctrller.init_controller(convert_type(
        freq), convert_type([stand_kp, stand_kd, joint_kp, joint_kd]))

    for _ in range(40):
        # p.setTimeStep(0.0015)
        p.stepSimulation()
        imu_data, leg_data, _ = get_data_from_sim()
        cpp_gait_ctrller.pre_work(convert_type(
            imu_data), convert_type(leg_data))

    for j in range(16):
        force = 0
        p.setJointMotorControl2(
            boxId, j, p.VELOCITY_CONTROL, force=force)


def init_simulator():
    global boxId, reset, low_energy_mode, high_performance_mode, terrain
    robot_start_pos = [0, 0, 0.42]
    p.connect(p.GUI)  # or p.DIRECT for non-graphical version
    p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
    reset = p.addUserDebugParameter("reset", 1, 0, 0)
    low_energy_mode = p.addUserDebugParameter("low_energy_mode", 1, 0, 0)
    high_performance_mode = p.addUserDebugParameter("high_performance_mode", 1, 0, 0)
    p.setGravity(0, 0, -9.8)
    p.setTimeStep(1.0/freq)
    p.resetDebugVisualizerCamera(0.2, 45, -30, [1, -1, 1])
    # p.setPhysicsEngineParameter(numSolverIterations=30)
    # p.setPhysicsEngineParameter(enableConeFriction=0)
    # planeId = p.loadURDF("plane.urdf")
    # p.changeDynamics(planeId, -1, lateralFriction=1.0)
    # boxId = p.loadURDF("/home/wgx/Workspace_Ros/src/cloudrobot/src/quadruped_robot.urdf", robot_start_pos,
    #                    useFixedBase=FixedBase)

    heightPerturbationRange = 0.06
    numHeightfieldRows = 256
    numHeightfieldColumns = 256
    if terrain == "plane":
        planeShape = p.createCollisionShape(shapeType=p.GEOM_PLANE)
        ground_id = p.createMultiBody(0, planeShape)
        p.resetBasePositionAndOrientation(ground_id, [0, 0, 0], [0, 0, 0, 1])
    elif terrain == "random1":
        heightfieldData = [0]*numHeightfieldRows*numHeightfieldColumns
        for j in range(int(numHeightfieldColumns/2)):
            for i in range(int(numHeightfieldRows/2)):
                height = random.uniform(0, heightPerturbationRange)
                heightfieldData[2*i+2*j*numHeightfieldRows] = height
                heightfieldData[2*i+1+2*j*numHeightfieldRows] = height
                heightfieldData[2*i+(2*j+1)*numHeightfieldRows] = height
                heightfieldData[2*i+1+(2*j+1)*numHeightfieldRows] = height
        terrainShape = p.createCollisionShape(shapeType=p.GEOM_HEIGHTFIELD, meshScale=[.05, .05, 1], heightfieldTextureScaling=(
            numHeightfieldRows-1)/2, heightfieldData=heightfieldData, numHeightfieldRows=numHeightfieldRows, numHeightfieldColumns=numHeightfieldColumns)
        ground_id = p.createMultiBody(0, terrainShape)
        p.resetBasePositionAndOrientation(ground_id, [0, 0, 0], [0, 0, 0, 1])
    elif terrain == "random2":
        terrain_shape = p.createCollisionShape(
            shapeType=p.GEOM_HEIGHTFIELD,
            meshScale=[.5, .5, .5],
            fileName="heightmaps/ground0.txt",
            heightfieldTextureScaling=128)
        ground_id = p.createMultiBody(0, terrain_shape)
        textureId = p.loadTexture("hybrid_gait/quadruped_ctrl/model/grass.png")
        p.changeVisualShape(ground_id, -1, textureUniqueId=textureId)
        p.resetBasePositionAndOrientation(ground_id, [1, 0, 0.2], [0, 0, 0, 1])
    elif terrain == "stairs":
        planeShape = p.createCollisionShape(shapeType=p.GEOM_PLANE)
        ground_id = p.createMultiBody(0, planeShape)
        # p.resetBasePositionAndOrientation(ground_id, [0, 0, 0], [0, 0.0872, 0, 0.9962])
        p.resetBasePositionAndOrientation(ground_id, [0, 0, 0], [0, 0, 0, 1])
        # many box
        colSphereId = p.createCollisionShape(
            p.GEOM_BOX, halfExtents=[0.1, 0.4, 0.01])
        colSphereId1 = p.createCollisionShape(
            p.GEOM_BOX, halfExtents=[0.1, 0.4, 0.02])
        colSphereId2 = p.createCollisionShape(
            p.GEOM_BOX, halfExtents=[0.1, 0.4, 0.03])
        colSphereId3 = p.createCollisionShape(
            p.GEOM_BOX, halfExtents=[0.1, 0.4, 0.04])
        # colSphereId4 = p.createCollisionShape(
        #     p.GEOM_BOX, halfExtents=[0.03, 0.03, 0.03])
        p.createMultiBody(100, colSphereId, basePosition=[1.0, 1.0, 0.0])
        p.changeDynamics(colSphereId, -1, lateralFriction=lateralFriction, spinningFriction=spinningFriction)
        p.createMultiBody(100, colSphereId1, basePosition=[1.2, 1.0, 0.0])
        p.changeDynamics(colSphereId1, -1, lateralFriction=lateralFriction, spinningFriction=spinningFriction)
        p.createMultiBody(100, colSphereId2, basePosition=[1.4, 1.0, 0.0])
        p.changeDynamics(colSphereId2, -1, lateralFriction=lateralFriction, spinningFriction=spinningFriction)
        p.createMultiBody(100, colSphereId3, basePosition=[1.6, 1.0, 0.0])
        p.changeDynamics(colSphereId3, -1, lateralFriction=lateralFriction, spinningFriction=spinningFriction)
        # p.createMultiBody(10, colSphereId4, basePosition=[2.7, 1.0, 0.0])
        # p.changeDynamics(colSphereId4, -1, lateralFriction=0.5)

    p.changeDynamics(ground_id, -1, lateralFriction=lateralFriction)
    boxId = p.loadURDF("mini_cheetah/mini_cheetah.urdf", robot_start_pos,
                       useFixedBase=False)

    jointIds = []
    for j in range(p.getNumJoints(boxId)):
        p.getJointInfo(boxId, j)
        jointIds.append(j)

    # slope terrain
    # colSphereId = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.5, 0.5, 0.001])
    # colSphereId1 = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.2, 0.5, 0.06])
    # BoxId = p.createMultiBody(100, colSphereId, basePosition=[1.0, 1.0, 0.0])
    # BoxId = p.createMultiBody(100, colSphereId1, basePosition=[1.6, 1.0, 0.0])

    # stairs
    # colSphereId = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.1, 0.4, 0.02])
    # colSphereId1 = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.1, 0.4, 0.04])
    # colSphereId2 = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.1, 0.4, 0.06])
    # colSphereId3 = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.1, 0.4, 0.08])
    # colSphereId4 = p.createCollisionShape(p.GEOM_BOX, halfExtents=[1.0, 0.4, 0.1])
    # BoxId = p.createMultiBody(100, colSphereId, basePosition=[1.0, 1.0, 0.0])
    # BoxId = p.createMultiBody(100, colSphereId1, basePosition=[1.2, 1.0, 0.0])
    # BoxId = p.createMultiBody(100, colSphereId2, basePosition=[1.4, 1.0, 0.0])
    # BoxId = p.createMultiBody(100, colSphereId3, basePosition=[1.6, 1.0, 0.0])
    # BoxId = p.createMultiBody(100, colSphereId4, basePosition=[2.7, 1.0, 0.0])

    reset_robot()


def main():
    cnt = 0
    flag = 0
    dist = 0
    dist_x = 0
    dist_y = 0
    energy = 0
    force = 0
    last_pos = [0.0, 0.0]
    rate = 1/500.0
    reset_flag = p.readUserDebugParameter(reset)
    low_energy_flag = p.readUserDebugParameter(low_energy_mode)
    high_performance_flag = p.readUserDebugParameter(high_performance_mode)
    while 1:
        # check reset button state
        if(reset_flag < p.readUserDebugParameter(reset)):
            reset_flag = p.readUserDebugParameter(reset)
            print("reset the robot")
            cnt = 0
            reset_robot()
        if(low_energy_flag < p.readUserDebugParameter(low_energy_mode)):
            low_energy_flag = p.readUserDebugParameter(low_energy_mode)
            print("set robot to low energy mode")
            cpp_gait_ctrller.set_robot_mode(convert_type(1))
        if(high_performance_flag < p.readUserDebugParameter(high_performance_mode)):
            high_performance_flag = p.readUserDebugParameter(high_performance_mode)
            print("set robot to high performance mode")
            cpp_gait_ctrller.set_robot_mode(convert_type(0))
        # get data from simulator
        imu_data, leg_data, base_pos = get_data_from_sim()

        # call cpp function to calculate mpc tau
        tau = cpp_gait_ctrller.toque_calculator(convert_type(
            imu_data), convert_type(leg_data))
        if fix_vel_test:
            # act = [20,20,20,20,20,20,20,20,20] # stand
            # act = [20,0,10,5,15,15,15,15,15] # walk20
            # act = [16,0,8,4,12,12,12,12,12] # walk16
            # act = [12,0,6,3,9,9,9,9,9] # walk10
            # act = [20,0,10,10,0,10,10,10,10] # trot20
            # act = [16,0,8,8,0,8,8,8,8] # trot16
            # act = [14,0,7,7,0,7,7,7,7] # trot14
            # act = [10,0,5,5,0,5,5,5,5] # trot10
            # act = [8,0,4,4,0,4,4,4,4] # trot8
            act = [10,0,3,6,9,5,5,5,5] # gallop10
            cpp_gait_ctrller.set_gait_param(convert_type(act))
            vel = [2.0, 0.0, 0.0] # fix vel test
            if(cnt > 500*2):
                cpp_gait_ctrller.set_robot_vel(convert_type(vel))
                dist += np.sqrt((base_pos[0]-last_pos[0])**2 + (base_pos[1]-last_pos[1])**2)
                dist_x += np.abs(base_pos[0]-last_pos[0])
                dist_y += np.abs(base_pos[1]-last_pos[1])
                energy += get_energy_consumption(tau.contents.eff, leg_data[12:24])
                force += np.sum(np.array(tau.contents.eff)**2)
                # print(energy)
            last_pos[0] = base_pos[0]
            last_pos[1] = base_pos[1]
            # if(cnt % 500 == 0 and cnt >= 500*5 and cnt < 500*70):
            #     print(energy)
            if(cnt >= 500*12):
                # print("v_des = {:.2f}".format(vel[0]), 
                #       "v_avg = {:.6f}".format(dist/10), 
                #       "vx_avg = {:.6f}".format(dist_x/10), 
                #       "vy_avg = {:.6f}".format(dist_y/10), 
                #       "dist = {:.6f}".format(dist), 
                #       "energy = {:.6f}".format(energy/dist))
                # print("{:.2f}".format(vel[0]), "{:.6f}".format(energy/dist))
                print("{:.2f}".format(vel[0]), "{:.6f}".format(force))
                cnt = 0
                flag += 1
                dist = 0
                dist_x = 0
                dist_y = 0
                energy = 0
                force = 0
        else:
            if(cnt > 500*2):
                vel = get_target_vel((cnt-500*2)/1000)
                # get_velocity = p.getBaseVelocity(boxId)
                # vel = [vel[0], 0, 0]
                cpp_gait_ctrller.set_robot_vel(convert_type(vel))
                dist += np.sqrt((base_pos[0]-last_pos[0])**2 + (base_pos[1]-last_pos[1])**2)
                dist_x += np.abs(base_pos[0]-last_pos[0])
                dist_y += np.abs(base_pos[1]-last_pos[1])
                energy += get_energy_consumption(tau.contents.eff, leg_data[12:24])
                # print((cnt-500*2)/500, vel, energy)
                # print((cnt-500*2)/500, energy)
                print(vel)
            last_pos[0] = base_pos[0]
            last_pos[1] = base_pos[1]
            if(cnt >= 500*2+30*500*2):
                return
            # if(cnt % 500 == 0):
                # print("vx_des = {:.2f}".format(vel[0]), 
                #       "vy_des = {:.2f}".format(vel[1]), 
                #       "yaw_des = {:.2f}".format(vel[2]), 
                #       "v_avg = {:.6f}".format(dist/10), 
                #       "vx_avg = {:.6f}".format(dist_x/10), 
                #       "vy_avg = {:.6f}".format(dist_y/10), 
                #       "dist = {:.6f}".format(dist), 
                #       "energy = {:.6f}".format(energy/dist))
                # print("{:.2f}".format(vel[0]), "{:.6f}".format(energy/dist))
        
        # set tau to simulator
        p.setJointMotorControlArray(bodyUniqueId=boxId,
                                    jointIndices=motor_id_list,
                                    controlMode=p.TORQUE_CONTROL,
                                    forces=tau.contents.eff)

        # reset visual cam
        p.resetDebugVisualizerCamera(2.5, 45, -30, base_pos)

        cnt += 1
        if cnt > 99999999:
            cnt = 99999999
        p.stepSimulation()
        time.sleep(rate)


if __name__ == '__main__':


    with open('hybrid_gait/quadruped_ctrl/config/quadruped_ctrl_config.yaml') as f:
        quadruped_param = yaml.safe_load(f)
        params = quadruped_param['simulation']

    visualization = params['visualization']
    terrain = params['terrain']
    lateralFriction = params['lateralFriction']
    spinningFriction = params['spinningFriction']
    freq = params['freq']
    stand_kp = params['stand_kp']
    stand_kd = params['stand_kd']
    joint_kp = params['joint_kp']
    joint_kd = params['joint_kd']

    so_file = 'hybrid_gait/quadruped_ctrl/build/libquadruped_ctrl.so'
    cpp_gait_ctrller = ctypes.cdll.LoadLibrary(so_file)
    cpp_gait_ctrller.get_prf_foot_coor.restype = ctypes.c_double
    cpp_gait_ctrller.toque_calculator.restype = ctypes.POINTER(
        StructPointer)

    init_simulator()

    main()
