import gym
import random
import numpy as np
from gym import spaces
from gym.utils import seeding
from mpi4py import MPI

import hybrid_gait_task
import hybrid_gait_robot


class HybridGaitGym(gym.Env):

    def __init__(self, action_repeat=100):
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
                                            np.inf, shape=(14, ), dtype='float32')
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

        act[0] = act[0]*14.0+6.0  # horizon: 6-20
        for i in range(8):
            act[i+1] = act[i+1]*act[0]  # offset, duration: 0-horizon

        # act = np.array([20,0,10,5,15,15,15,15,15]) # walk20
        # act = np.array([16,0,8,4,12,12,12,12,12]) # walk16
        # act = np.array([20,0,10,10,0,10,10,10,10]) # trot20
        # act = np.array([16,0,8,8,0,8,8,8,8]) # trot16
        # act = np.array([12,0,6,6,0,6,6,6,6]) # trot12

        if(type(act) == type(np.array([1]))):
            act = act.tolist()
        if(type(act[0]) == np.float64):
            act = [act[j].item() for j in range(9)]
        act = [round(act[j]) for j in range(9)]
        if MPI.COMM_WORLD.Get_rank() == 0:
            print(act)

        self.robot.set_vel(self._get_target_vel())
        obs, safe = self.robot.step(act)

        rew = self.task.get_reward(obs, self.step_time)
        if not np.any(obs):
            rew = 0.0

        self.step_time += 1

        # done = self.task.get_done(obs)
        done = False
        if self.step_time > 1000:
            done = True
        done = not safe or done

        return obs, rew, done, {}

    def _get_target_vel(self):
        # t = self.step_time/5.0
        # nf = 5
        # w_f = 0.8
        # q0 = [0.0]
        # q = [0.0] * 3
        # qq = 0
        # if t == 0:
        #     self.a = [random.uniform(-1,1)  for _ in range(nf)]
        #     self.b = [random.uniform(-1,1)  for _ in range(nf)]
        # # a = [0.2886,-0.2428,0.6232,0.0657,-0.2985]
        # # b = [0.8780,0.7519,0.1003,0.2450,0.1741]

        # for l in range(2, nf):
        #     qq += (self.a[l-2]/(w_f*l))*np.sin(w_f*l*t) - (self.b[l-2]/(w_f*l))*np.cos(w_f*l*t)
        #     q0 += (self.a[l-2]/(w_f*l))*np.sin(w_f*l*0) - (self.b[l-2]/(w_f*l))*np.cos(w_f*l*0)
        # qq -= q0
        # q[0] = qq.tolist()[0]

        # if MPI.COMM_WORLD.Get_rank() == 0:
        #     print("vel = ", q)

        # return [1.2, 0, 0]
        if self.step_time%5.0 == 0:
            self.target_vel = [random.uniform(-1,1.5), 0.0, 0.0]
        if MPI.COMM_WORLD.Get_rank() == 0:
            print("vel = ", self.target_vel)
        return self.target_vel
        # return q
