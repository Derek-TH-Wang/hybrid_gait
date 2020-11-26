import numpy as np


class HybridGaitTask(object):
    """Hybrid Gait task."""

    def __init__(self,
                 weight=1.0,
                 balance_weight=0.3,
                 velocity_weight=0.6,
                 energy_weight=0.1,
                 balance_scale=1,
                 velocity_err_scale=30,
                 energy_scale=10):
        
        self._weight = weight

        # reward function parameters
        self._balance_weight = balance_weight
        self._velocity_weight = velocity_weight
        self._energy_weight = energy_weight

        self._balance_scale = balance_scale
        self._velocity_err_scale = velocity_err_scale
        self._energy_scale = energy_scale

        return


    def get_done(self, obs):
        vel_follow_fail = False
        diff_vel = obs[0:3]
        diff_yaw_rate = obs[3]
        if abs(diff_vel[0]) > 0.3 or \
           abs(diff_vel[1]) > 0.2 or \
           abs(diff_vel[2]) > 0.1 or \
           abs(diff_yaw_rate > 0.1):
            vel_follow_fail = True

        done = vel_follow_fail

        return done


    def get_reward(self, obs):
        """Get the reward without side effects."""

        velocity_reward = self._calc_reward_velocity(obs[0:4])
        balance_reward = self._calc_reward_balance(obs[4:7], obs[7:9], obs[9:11])
        energy_reward = self._calc_reward_energy(obs[11])
        # end_effector_reward = self._calc_reward_end_effector()

        reward = self._balance_weight * balance_reward \
            + self._velocity_weight * velocity_reward \
            + self._energy_weight * energy_reward
        print("rew = ", balance_reward, velocity_reward, energy_reward, reward)
        return reward * self._weight


    def _calc_reward_velocity(self, vel_diff):
        vel_err = vel_diff.dot(vel_diff)

        velocity_reward = np.exp(-self._velocity_err_scale * vel_err)
        return velocity_reward


    def _calc_reward_balance(self, acc, rpy, rpy_rate):
        acc_err = acc.dot(acc)
        rpy_err = rpy.dot(rpy)
        rpy_rate_err = rpy_rate.dot(rpy_rate)
        balance_err = acc_err*0.4 + rpy_err*0.3 + rpy_rate_err*0.3
        
        balance_reward = np.exp(-self._balance_scale * balance_err)
        return balance_reward


    def _calc_reward_energy(self, energy_consumption):
        energy_reward = np.exp(-self._energy_scale * energy_consumption)
        return energy_reward
