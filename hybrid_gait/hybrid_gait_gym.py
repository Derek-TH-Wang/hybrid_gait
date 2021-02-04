import gym
import random
import numpy as np
from gym import spaces
from gym.utils import seeding
from mpi4py import MPI

import hybrid_gait_task
import hybrid_gait_robot


class HybridGaitGym(gym.Env):

    def __init__(self, action_repeat=400):
        self.seed()
        self.action_repeat = action_repeat
        self.robot = hybrid_gait_robot.HybridGaitRobot(self.action_repeat)
        self.task = hybrid_gait_task.HybridGaitTask()

        self.step_time = 0
        self.target_vel = [0] * 3

        self.action_space = spaces.Box(
            np.array([0.0]*9),
            np.array([1.0]*9),
            dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf,
                                            np.inf, shape=(16, ), dtype='float32')
        self.reset()

        return

    def seed(self, seed=None):
        self.np_random, self.np_random_seed = seeding.np_random(seed)
        return [self.np_random_seed]

    def reset(self):
        self.step_time = 0
        obs = self.robot.reset_robot()
        return obs

    def step(self, action):
        act = np.zeros(9)
        for i in range(9):  # sigmoid
            act[i] = 1.0/(1.0 + np.exp(-action[i]))

        act[0] = act[0]*16.0+4.0  # horizon: 4-20
        for i in range(8):
            act[i+1] = act[i+1]*act[0]  # offset, duration: 0-horizon

        # act = np.array([20,20,20,20,20,20,20,20,20]) # stand
        # act = np.array([24,0,12,6,18,18,18,18,18]) # 24
        # act = np.array([20,0,10,5,15,15,15,15,15]) # walk20
        # act = np.array([16,0,8,4,12,12,12,12,12]) # walk16
        # act = np.array([16,0,8,8,0,12,12,12,12]) # ???
        # act = np.array([12,0,6,3,9,9,9,9,9]) # walk10
        # act = np.array([20,0,10,10,0,10,10,10,10]) # trot20
        # act = np.array([16,0,8,8,0,8,8,8,8]) # trot16
        # act = np.array([14,0,7,7,0,7,7,7,7]) # trot14
        # act = np.array([10,0,5,5,0,5,5,5,5]) # trot10
        # act = np.array([8,0,4,4,0,4,4,4,4]) # trot8
        # act = np.array([10,0,2,7,9,5,5,5,5]) # gallop10
        # act = np.array([8,0,2,5,8,4,4,4,4]) # gallop8
        # act = np.array([6,0,4,6,0,5,3,2,5]) #v0.05
        # act = np.array([5,0,3,4,0,3,2,2,3]) #v1.0

        if(type(act) == type(np.array([1]))):
            act = act.tolist()
        if(type(act[0]) == np.float64):
            act = [act[j].item() for j in range(9)]
        act = [round(act[j]) for j in range(9)]
        if MPI.COMM_WORLD.Get_rank() == 0:
            print(act)

        self.robot.set_vel(self._get_target_vel())
        obs, safe = self.robot.step(act)

        rew = self.task.get_reward(obs)

        self.step_time += 1

        # done = self.task.get_done(obs)
        done = False
        if self.step_time > 1000:
            done = True
        done = not safe or done

        return obs, rew, done, {}

    def _get_target_vel(self):
        # t = self.step_time/10
        # nf = 5
        # w_f = 0.8
        # q = [0.0] * 3

        # q0 = [0.0]
        # qq = 0
        # if t == 0:
        #     self.ax = [random.uniform(-1,1)  for _ in range(nf)]
        #     self.bx = [random.uniform(-1,1)  for _ in range(nf)]

        # for l in range(2, nf):
        #     qq += (self.ax[l-2]/(w_f*l))*np.sin(w_f*l*t) - (self.bx[l-2]/(w_f*l))*np.cos(w_f*l*t)
        #     q0 += (self.ax[l-2]/(w_f*l))*np.sin(w_f*l*0) - (self.bx[l-2]/(w_f*l))*np.cos(w_f*l*0)
        # # qq -= q0
        # # q[0] = qq.tolist()[0]
        # q[0] = qq.tolist()

        # q0 = [0.0]
        # qq = 0
        # if t == 0:
        #     self.ay = [random.uniform(-1,1)  for _ in range(nf)]
        #     self.by = [random.uniform(-1,1)  for _ in range(nf)]

        # for l in range(2, nf):
        #     qq += (self.ay[l-2]/(w_f*l))*np.sin(w_f*l*t) - (self.by[l-2]/(w_f*l))*np.cos(w_f*l*t)
        #     q0 += (self.ay[l-2]/(w_f*l))*np.sin(w_f*l*0) - (self.by[l-2]/(w_f*l))*np.cos(w_f*l*0)
        # # qq -= q0
        # # q[1] = qq.tolist()[0]
        # q[1] = qq.tolist()

        # q0 = [0.0]
        # qq = 0
        # if t == 0:
        #     self.aw = [random.uniform(-1,1)  for _ in range(nf)]
        #     self.bw = [random.uniform(-1,1)  for _ in range(nf)]

        # for l in range(2, nf):
        #     qq += (self.aw[l-2]/(w_f*l))*np.sin(w_f*l*t) - (self.bw[l-2]/(w_f*l))*np.cos(w_f*l*t)
        #     q0 += (self.aw[l-2]/(w_f*l))*np.sin(w_f*l*0) - (self.bw[l-2]/(w_f*l))*np.cos(w_f*l*0)
        # # qq -= q0
        # # q[2] = qq.tolist()[0]
        # q[2] = qq.tolist()

        # q = [0.1, 0, 0]
        # # if MPI.COMM_WORLD.Get_rank() == 0:
        # #     print("vel = ", q)
        # return q

        # return [1.2, 0, 0]
        if self.step_time == 0:
            # self.target_vel = [random.uniform(-1,1.5), random.uniform(-0.5,0.5), random.uniform(-0.5,0.5)]
            self.target_vel = [random.uniform(-1.5,2.5), 0.0, 0.0]
            # self.target_vel = [1.0, 0.0, 0.0]
        if MPI.COMM_WORLD.Get_rank() == 0:
            print("vel = ", self.target_vel)
        return self.target_vel
