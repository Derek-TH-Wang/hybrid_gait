import numpy as np
from mpi4py import MPI


class HybridGaitTask(object):
    """Hybrid Gait task."""

    def __init__(self,
                 weight=1.0,
                 velocity_weight=0.2,
                 balance_weight=0.25,
                 energy_weight=0.5,
                 time_weight=0.05,
                 velocity_err_scale=5,
                 balance_scale=1,
                 energy_scale=20,
                 time_scale = 0.001):

        self._weight = weight

        # reward function parameters
        self._velocity_weight = velocity_weight
        self._balance_weight = balance_weight
        self._energy_weight = energy_weight
        self._time_weight = time_weight

        self._velocity_err_scale = velocity_err_scale
        self._balance_scale = balance_scale
        self._energy_scale = energy_scale
        self._time_scale = time_scale

        return

    # def get_done(self, obs):
    #     vel_follow_fail = False
    #     diff_vel = obs[0:3]
    #     diff_yaw_rate = obs[3]
    #     if abs(diff_vel[0]) > 0.3 or \
    #        abs(diff_vel[1]) > 0.2 or \
    #        abs(diff_vel[2]) > 0.1 or \
    #        abs(diff_yaw_rate > 0.1):
    #         vel_follow_fail = True

    #     done = vel_follow_fail

    #     return done

    def get_reward(self, obs, step_time):
        """Get the reward without side effects."""

        velocity_reward = self._calc_reward_velocity(obs[0:3])
        balance_reward = self._calc_reward_balance(
            obs[3:6], obs[6:8], obs[8:10])
        energy_reward = self._calc_reward_energy(obs[10])
        time_reward = self._calc_reward_time(step_time)

        reward = self._balance_weight * balance_reward \
            + self._velocity_weight * velocity_reward \
            + self._energy_weight * energy_reward \
            + self._time_weight * time_reward

        reward = pow(reward, 0.5) * self._weight

        if MPI.COMM_WORLD.Get_rank() == 0:
            print("rew = {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}".format(velocity_reward, balance_reward, energy_reward, time_reward, reward))
            # print("{:.6f}".format(reward))
        return reward

    def _calc_reward_velocity(self, vel_diff):
        vel_err = np.sqrt(vel_diff.dot(vel_diff))

        velocity_reward = np.exp(-self._velocity_err_scale * vel_err)
        return velocity_reward

    def _calc_reward_balance(self, acc, rpy, rpy_rate):
        acc_err = np.sqrt(acc.dot(acc))
        rpy_err = np.sqrt(rpy.dot(rpy*180.0/np.pi))
        rpy_rate_err = np.sqrt(rpy_rate.dot(rpy_rate))
        balance_err = acc_err*0.1 + rpy_err*0.6 + rpy_rate_err*0.3
        # print("{:.6f} {:.6f} {:.6f}".format(acc_err*0.1, rpy_err*0.6, rpy_rate_err*0.3))
        balance_reward = np.exp(-self._balance_scale * balance_err)
        return balance_reward

    def _calc_reward_energy(self, energy_consumption):
        energy_reward = np.exp(-self._energy_scale * energy_consumption)
        return energy_reward

    def _calc_reward_time(self, step_time):
        return step_time*self._time_scale