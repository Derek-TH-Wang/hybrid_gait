import gym
import numpy as np
from gym.utils import seeding

from hybrid_gait import hybrid_gait_task
from hybrid_gait.quadruped_ctrl.scripts import hybrid_gait_robot


class HybridGaitGym(gym.Env):

    def __init__(self, action_repeat=500):
        self.seed()
        self.action_repeat = action_repeat
        self.robot = hybrid_gait_robot.HybridGaitRobot(self.action_repeat)
        self.task = hybrid_gait_task.HybridGaitTask()


    def seed(self, seed=None):
        self.np_random, self.np_random_seed = seeding.np_random(seed)
        return [self.np_random_seed]


    def reset(self):
        obs = self.robot.reset_robot()
        return obs


    def step(self, action):
        act = [0]*9
        act[0] = round(action[0]*12.0+8.0) # horizon: 8-20
        for i in range(8):
            act[i+1] = round(action[i+1]*act[0]) # offset, duration: 0-horizon
        self.robot.set_vel(self._get_target_vel())
        safe, obs = self.robot.step(act)
        rew = self.task.get_reward(obs)
        done = self.task.get_done(obs)

        done = not safe or done

        return obs, rew, done, {}

    
    def _get_target_vel(self):
        # t = self._state_action_counter/1000
        # nf = 5
        # w_f = 0.8
        # q0 = [0.0]
        # q = [0.0] * self._num_cmd
        # qq = 0
        # # if t == 0:
        # #     a = [random.uniform(-1,1)  for _ in range(nf)]
        # #     b = [random.uniform(-1,1)  for _ in range(nf)]
        # a = [0.2886,-0.2428,0.6232,0.0657,-0.2985]
        # b = [0.8780,0.7519,0.1003,0.2450,0.1741]

        # for l in range(2, nf):
        #     qq += (a[l-2]/(w_f*l))*np.sin(w_f*l*t) - (b[l-2]/(w_f*l))*np.cos(w_f*l*t)
        #     q0 += (a[l-2]/(w_f*l))*np.sin(w_f*l*0) - (b[l-2]/(w_f*l))*np.cos(w_f*l*0)
        # qq -= q0
        # q[0] = qq.tolist()[0]/2
        
        return [0.25, 0, 0]