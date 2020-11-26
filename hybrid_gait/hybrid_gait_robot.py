import yaml
import numpy as np
import ctypes
import pybullet as p
import pybullet_data


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


class StructPointer(ctypes.Structure):
    _fields_ = [("eff", ctypes.c_double * 12)]


class HybridGaitRobot(object):

    def __init__(self, action_repeat=1):

        self.action_repeat = action_repeat

        self.ground = 0
        self.quadruped = 0

        self.init_pos = [0, 0, 0.3]
        self.motor_id_list = [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14]
        self.init_new_pos = [0.0, -0.8, 1.6, 0.0, -0.8, 1.6, 0.0, -0.8, 1.6, 0.0, -0.8, 1.6,
                             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        self.tau = [0]*12
        self.imu_data = [0]*10
        self.leg_data = self.init_new_pos
        self.target_base_vel = [0]*3
        self.base_vel = [0]*3
        self.base_position = self.init_pos
        self._last_base_pos = self.init_pos
        self._robot_dist = 0
        self.get_last_vel = [0]*3

        with open('hybrid_gait/quadruped_ctrl/config/quadruped_ctrl.yaml') as f:
            quadruped_param = yaml.safe_load(f)
            params = quadruped_param['simulation']

        self.terrain = params['terrain']
        self.lateralFriction = params['/lateralFriction']
        self.spinningFriction = params['/spinningFriction']
        self.freq = params['/freq']
        self.stand_kp = params['/stand_kp']
        self.stand_kd = params['/stand_kd']
        self.joint_kp = params['/joint_kp']
        self.joint_kd = params['/joint_kd']

        so_file = 'hybrid_gait/quadruped_ctrl/build/libquadruped_ctrl.so'
        self.cpp_gait_ctrller = ctypes.cdll.LoadLibrary(so_file)
        self.cpp_gait_ctrller.toque_calculator.restype = ctypes.POINTER(StructPointer)

        self.init_simulator()


    def init_simulator(self):
        p.connect(p.GUI)  # or p.DIRECT for non-graphical version
        p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
        p.setGravity(0, 0, -9.8)
        p.setTimeStep(1.0/self.freq)
        p.resetDebugVisualizerCamera(0.2, 45, -30, [1, -1, 1])

        heightPerturbationRange = 0.06
        numHeightfieldRows = 256
        numHeightfieldColumns = 256
        if self.terrain == "plane":
            planeShape = p.createCollisionShape(shapeType=p.GEOM_PLANE)
            self.ground = p.createMultiBody(0, planeShape)
            p.resetBasePositionAndOrientation(self.ground, [0, 0, 0], [0, 0, 0, 1])
        elif self.terrain == "random1":
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
            self.ground = p.createMultiBody(0, terrainShape)
            p.resetBasePositionAndOrientation(self.ground, [0, 0, 0], [0, 0, 0, 1])
        elif self.terrain == "random2":
            terrain_shape = p.createCollisionShape(
                shapeType=p.GEOM_HEIGHTFIELD,
                meshScale=[.5, .5, .5],
                fileName="heightmaps/ground0.txt",
                heightfieldTextureScaling=128)
            self.ground = p.createMultiBody(0, terrain_shape)
            textureId = p.loadTexture("hybrid_gait/quadruped_ctrl/model/grass.png")
            p.changeVisualShape(self.ground, -1, textureUniqueId=textureId)
            p.resetBasePositionAndOrientation(self.ground, [1, 0, 0.2], [0, 0, 0, 1])
        elif self.terrain == "stairs":
            planeShape = p.createCollisionShape(shapeType=p.GEOM_PLANE)
            self.ground = p.createMultiBody(0, planeShape)
            # p.resetBasePositionAndOrientation(self.ground, [0, 0, 0], [0, 0.0872, 0, 0.9962])
            p.resetBasePositionAndOrientation(self.ground, [0, 0, 0], [0, 0, 0, 1])
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
            p.changeDynamics(colSphereId, -1, lateralFriction=self.lateralFriction)
            p.createMultiBody(100, colSphereId1, basePosition=[1.2, 1.0, 0.0])
            p.changeDynamics(colSphereId1, -1, lateralFriction=self.lateralFriction)
            p.createMultiBody(100, colSphereId2, basePosition=[1.4, 1.0, 0.0])
            p.changeDynamics(colSphereId2, -1, lateralFriction=self.lateralFriction)
            p.createMultiBody(100, colSphereId3, basePosition=[1.6, 1.0, 0.0])
            p.changeDynamics(colSphereId3, -1, lateralFriction=self.lateralFriction)
            # p.createMultiBody(10, colSphereId4, basePosition=[2.7, 1.0, 0.0])
            # p.changeDynamics(colSphereId4, -1, lateralFriction=0.5)

        p.changeDynamics(self.ground, -1, lateralFriction=self.lateralFriction)
        self.quadruped = p.loadURDF("mini_cheetah/mini_cheetah.urdf", self.init_pos,
                        useFixedBase=False)
        p.changeDynamics(self.quadruped, 3, spinningFriction=self.spinningFriction)
        p.changeDynamics(self.quadruped, 7, spinningFriction=self.spinningFriction)
        p.changeDynamics(self.quadruped, 11, spinningFriction=self.spinningFriction)
        p.changeDynamics(self.quadruped, 15, spinningFriction=self.spinningFriction)
        jointIds = []
        for j in range(p.getNumJoints(self.quadruped)):
            p.getJointInfo(self.quadruped, j)
            jointIds.append(j)


    def reset_robot(self):
        p.resetBasePositionAndOrientation(
            self.quadruped, self.init_pos, [0, 0, 0, 1])
        for j in range(12):
            p.resetJointState(self.quadruped, self.motor_id_list[j], self.init_new_pos[j])
        self.cpp_gait_ctrller.init_controller(convert_type(
            self.freq), convert_type([self.stand_kp, self.stand_kd, self.joint_kp, self.joint_kd]))

        for _ in range(40):
            p.stepSimulation()
            self._get_data_from_sim()
            self.cpp_gait_ctrller.pre_work(convert_type(
                self.imu_data), convert_type(self.leg_data))

        for j in range(16):
            force = 0
            p.setJointMotorControl2(
                self.quadruped, j, p.VELOCITY_CONTROL, force=force)


    def step(self, gait_param):
        obs = np.array([0]*12)
        self._robot_dist = 0

        self.cpp_gait_ctrller.set_gait_param(convert_type(gait_param))
        for _ in range(self.action_repeat):
            self._run()
            obs = self._get_obs(obs)

        return obs


    def set_vel(self, target_base_vel):
        self.target_base_vel = target_base_vel
        self.cpp_gait_ctrller.set_robot_vel(convert_type(self.target_base_vel))


    def _cal_energy_consumption(self):
        engergy = 0
        for i in range(12):
            engergy += np.abs(self.tau[i] * self.leg_data[12+i]) * (1.0 / self.freq)
        engergy /= 1000.0
        self._robot_dist += ((self.base_position[0] - self._last_base_pos[0])**2 + 
                            (self.base_position[1] - self._last_base_pos[1])**2)
        self._last_base_pos = self.base_position
        engergy_consumption = engergy / self._robot_dist

        return engergy_consumption.item()


    def _get_obs(self, obs):
        rpy = p.getEulerFromQuaternion(self.imu_data[3:7])
        rpy_rate = self.imu_data[7:10]
        energy = self._cal_energy_consumption()

        obs[0:3] += np.abs(np.array(self.target_base_vel) - np.array(self.base_vel)) # linear xyz vel
        obs[3] += np.abs(self.target_base_vel[3] - self.imu_data[9]) # angular z vel
        obs[4:7] += np.abs(np.array(self.imu_data[0:3])) # linear acc
        obs[7:9] += np.abs(np.array(rpy[0:2]))
        obs[9:11] += np.abs(np.array(rpy_rate[0:2]))
        obs[12] += np.abs(np.array(energy))
        # obs = obs.tolist()

        return obs


    def _run(self):
        # get data from simulator
        self._get_data_from_sim()

        # call cpp function to calculate mpc tau
        tau = self.cpp_gait_ctrller.toque_calculator(convert_type(
            self.imu_data), convert_type(self.leg_data))
        
        for i in range(12):
            self.tau[i] = tau.contents.eff[i]

        # set tau to simulator
        p.setJointMotorControlArray(bodyUniqueId=self.quadruped,
                                    jointIndices=self.motor_id_list,
                                    controlMode=p.TORQUE_CONTROL,
                                    forces=self.tau)

        # reset visual cam
        p.resetDebugVisualizerCamera(2.5, 45, -30, self.base_position)

        p.stepSimulation()

        return


    def _get_data_from_sim(self):
        get_matrix = []
        get_velocity = []
        get_invert = []

        self.base_position, base_orientation = p.getBasePositionAndOrientation(self.quadruped)
        get_velocity = p.getBaseVelocity(self.quadruped)

        get_invert = p.invertTransform(self.base_position, base_orientation)
        get_matrix = p.getMatrixFromQuaternion(get_invert[1])

        # IMU ori
        self.imu_data[3] = base_orientation[0]
        self.imu_data[4] = base_orientation[1]
        self.imu_data[5] = base_orientation[2]
        self.imu_data[6] = base_orientation[3]
        # IMU rpy_rate
        self.imu_data[7] = get_matrix[0] * get_velocity[1][0] + get_matrix[1] * \
            get_velocity[1][1] + get_matrix[2] * get_velocity[1][2]
        self.imu_data[8] = get_matrix[3] * get_velocity[1][0] + get_matrix[4] * \
            get_velocity[1][1] + get_matrix[5] * get_velocity[1][2]
        self.imu_data[9] = get_matrix[6] * get_velocity[1][0] + get_matrix[7] * \
            get_velocity[1][1] + get_matrix[8] * get_velocity[1][2]
        # IMU acc
        linear_X = (get_velocity[0][0] - self.get_last_vel[0]) * self.freq
        linear_Y = (get_velocity[0][1] - self.get_last_vel[1]) * self.freq
        linear_Z = 9.8 + (get_velocity[0][2] - self.get_last_vel[2]) * self.freq
        self.imu_data[0] = get_matrix[0] * linear_X + \
            get_matrix[1] * linear_Y + get_matrix[2] * linear_Z
        self.imu_data[1] = get_matrix[3] * linear_X + \
            get_matrix[4] * linear_Y + get_matrix[5] * linear_Z
        self.imu_data[2] = get_matrix[6] * linear_X + \
            get_matrix[7] * linear_Y + get_matrix[8] * linear_Z

        self.base_vel[0] = get_matrix[0] * get_velocity[0][0] + get_matrix[1] * \
            get_velocity[0][1] + get_matrix[2] * get_velocity[0][2]
        self.base_vel[1] = get_matrix[3] * get_velocity[0][0] + get_matrix[4] * \
            get_velocity[0][1] + get_matrix[5] * get_velocity[0][2]
        self.base_vel[2] = get_matrix[6] * get_velocity[0][0] + get_matrix[7] * \
            get_velocity[0][1] + get_matrix[8] * get_velocity[0][2]

        # joint data
        joint_state = p.getJointStates(self.quadruped, self.motor_id_list)
        self.leg_data[0:12] = [joint_state[0][0], joint_state[1][0], joint_state[2][0],
                        joint_state[3][0], joint_state[4][0], joint_state[5][0],
                        joint_state[6][0], joint_state[7][0], joint_state[8][0],
                        joint_state[9][0], joint_state[10][0], joint_state[11][0]]

        self.leg_data[12:24] = [joint_state[0][1], joint_state[1][1], joint_state[2][1],
                        joint_state[3][1], joint_state[4][1], joint_state[5][1],
                        joint_state[6][1], joint_state[7][1], joint_state[8][1],
                        joint_state[9][1], joint_state[10][1], joint_state[11][1]]
        com_velocity = [get_velocity[0][0],
                        get_velocity[0][1], get_velocity[0][2]]
        self.get_last_vel.clear()
        self.get_last_vel = com_velocity

        return

