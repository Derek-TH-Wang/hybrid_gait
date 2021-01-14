import numpy as np
from mpi4py import MPI


class HybridGaitTask(object):
    """Hybrid Gait task."""

    def __init__(self,
                 weight=1.0,
                 velocity_weight=0.1,
                 balance_weight=0.5,
                 energy_weight=0.3,
                 time_weight=0.1,
                 velocity_err_scale=5,
                 balance_scale=1,
                 energy_scale=15,
                 time_scale = 1):

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

    def get_reward(self, obs):
        """Get the reward without side effects."""
        if not any(obs):
            return 0.0

        velocity_reward = self._calc_reward_velocity(obs[0:6])
        balance_reward = self._calc_reward_balance(
            obs[6:9], obs[9:11], obs[11:13], obs[13])
        energy_reward = self._calc_reward_energy(obs[14])
        time_reward = self._calc_reward_time(obs[15])

        vel = np.sqrt(obs[0]**2 + obs[1]**2)
        self._balance_weight = 0.8-0.45*np.log(vel+1)
        self._energy_weight = 0.45*np.log(vel+1)

        reward = self._balance_weight * balance_reward \
            + self._velocity_weight * velocity_reward \
            + self._energy_weight * energy_reward \
            + self._time_weight * time_reward

        reward = reward * self._weight

        if MPI.COMM_WORLD.Get_rank() == 0:
            print("rew = {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}".format(velocity_reward, balance_reward, energy_reward, time_reward, reward))
            # print("{:.6f}".format(reward))
        return reward

    def _calc_reward_velocity(self, vel):
        vel_diff = vel[0:3] - vel[3:6]
        vel_err = np.sqrt(vel_diff.dot(vel_diff))

        velocity_reward = np.exp(-self._velocity_err_scale * vel_err)
        return velocity_reward

    def _calc_reward_balance(self, acc, rpy, rpy_rate, foot_coor):
        acc_err = np.sqrt(acc.dot(acc))
        rpy_err = np.sqrt(rpy.dot(rpy*180.0/np.pi))
        rpy_rate_err = np.sqrt(rpy_rate.dot(rpy_rate))
        center_g_err = 100.0*foot_coor
        balance_err = acc_err*0.1 + rpy_err*0.1 + rpy_rate_err*0.1 + center_g_err*0.7
        # print("{:.6f} {:.6f} {:.6f}".format(acc_err*0.1, rpy_err*0.6, rpy_rate_err*0.3))
        balance_reward = np.exp(-self._balance_scale * balance_err)
        return balance_reward

    def _calc_reward_energy(self, energy_consumption):
        energy_reward = np.exp(-self._energy_scale * energy_consumption)
        return energy_reward

    def _calc_reward_time(self, step_time):
        return step_time*self._time_scale