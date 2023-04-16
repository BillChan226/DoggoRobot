# from safety_gym import safety_gym
import sys

sys.path.append("/home/gong112/service_backup/work/zhaorun/doggo/safety-gym/")
import safety_gym
import gym
import gym.envs.registration as r
import os
import math
import time
# print(r)

import numpy as np


class DoggoController:
    def __init__(self):
        self.phase_offsets = [0, math.pi / 2, math.pi, 3 * math.pi / 2]
        self.amplitude = 0.2  # adjust this to control speed
        self.frequency = 2  # adjust this to control stride frequency
        # self.hip_z_phase = [0.5, -0.5, 0.5, -0.5]

        self.left = [0.1, 0, 0.1, 0]
        self.right = [0, 0.1, 0, 0.1]
        # self.hip_z_phase = [0.1, 0.1, 0.1, 0.1]
        self.hip_y_phase = [0.1, 0.1, 0.1, 0.1]
        self.vel_pre = np.zeros(3)
        self.vel_start = 0
        self.flag = False

        self.kp = 1.5
        self.ki = 0
        self.kd = 0

    def get_actions(self, observation):

        action = np.zeros(12)
        vel_new = obs[101:104]

        if self.flag == False:
            self.flag = True
            self.vel_start = vel_new[0]

        if vel_new[0] - self.vel_start > 0:
            hip_z_phase = self.left
        else:
            hip_z_phase = self.right

        self.vel_pre = vel_new

        for i in range(4):
            # dir_z = observation[48 + 4 * i] - hip_z_phase[i]

            action[i] = 1

            # dir_y = observation[46 + 4 * i]  # + self.hip_y_phase[i]

            action[i + 4] = 1

            action[i + 8] = 0

            # error1 = observation[48+4*i]-0.1
            # error2 = observation[50+4*i]-0

            # if abs(error1) > 0.1:

            #     action[i] = -np.clip(error1*1.5, -1, 1)
            #     if i == 0: action[i] = action[i] #+ 0.3
            # else:
            #     action[i] = 0

            # if abs(error2) > 0.1:
            #     action[i+4] = -np.clip(error2, -1, 1)
            # else: action[i+4] = 0

        # action[i+8] = 0
        # print("hip_y", observation[46+4*i])
        # print("hip_z", observation[48+4*i])
        # print("ankle1", observation[38+4*i])
        # print("ankle2", observation[39+4*i])

        # action[i+8] = 0 # match the hip motion for now
        # print("error1", error1)

        return action


if __name__ == "__main__":

    env = gym.make('Safexp-DoggoGoal2-v0')
    obs = env.reset()
    # model = PIDController()
    model = DoggoController()

    start_time = time.time()
    action = np.zeros(12)

    for i in range(1000):
        # print("obs", obs)
        # for sensor in env.sensors_obs:  # Explicitly listed sensors
        #         dim = env.robot.sensor_dim[sensor]
        #         print("obs-dim",sensor, dim)
        # action = np.ones(12)
        # print("action", action)
        action = np.zeros(12)
        action[11] = -1
        action[7] = -1
        print(1)
        # action 4 hip1y
        # action 8 hip4y
        # for i in range(3):
        #     # dir_z = observation[48 + 4 * i] - hip_z_phase[i]
        #
        #     action[i] = 1
        #
        #     # dir_y = observation[46 + 4 * i]  # + self.hip_y_phase[i]
        #
        #     action[i + 4] = 1
        #
        #     action[i + 8] = 1
        # for i in [3]:
        #     # dir_z = observation[48 + 4 * i] - hip_z_phase[i]
        #
        #     action[i] = -1
        #
        #     # dir_y = observation[46 + 4 * i]  # + self.hip_y_phase[i]
        #
        #     action[i + 4] = -1
        #
        #     action[i + 8] = -1

        obs, reward, done, info = env.step(action)
        time.sleep(1)
        env.render()

        # print("velocimeter", obs[101:104])
