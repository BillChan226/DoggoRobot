# # from safety_gym import safety_gym
# import sys
#
# # sys.path.append("/home/gong112/service_backup/work/zhaorun/doggo/safety-gym/")
# import safety_gym
# import gym
# import gym.envs.registration as r
# import os
# import math
# import time
# import matplotlib.pyplot as plt
# # print(r)

import numpy as np


class DoggoController:
    def __init__(self):
        # self.ankle_p = 3
        # self.ankle_i = 0.2
        # self.ankle_d = 0.5
        # self.hip_y_p = 3
        # self.hip_y_i = 0.3
        # self.hip_y_d = 0.5
        # self.hip_z_p = 1.5
        # self.hip_z_i = 0.1
        # self.hip_Z_d = 0.5
        self.old_error = np.zeros(12)

        self.cumulative_error = np.zeros(12)
        self.kp = np.zeros(12)
        self.ki = np.zeros(12)
        self.kd = np.zeros(12)
        self.kp = np.array([0.4, 0.4, 0.4, 0.4, 1.2, 1.2, 1.2, 1.2, 1.5, 1.5, 1.5, 1.5])
        self.ki = np.array([0.0005, 0.0005, 0.0005, 0.0005, 0.015, 0.015, 0.015, 0.015, 0.008, 0.008, 0.008, 0.008])
        self.kd = np.array([-0.06, -0.06, -0.06, -0.06, -0.15, -0.15, -0.15, -0.15, -0.3, -0.3, -0.3, -0.3])
        # self.kp[4] = 1.2
        # self.ki[4] = 0.015
        # self.kd[4] = -0.15
        # self.kp[0] = 0.4
        # self.ki[0] = 0.0005
        # self.kd[0] = -0.06
        # self.kp[8] = 1.5
        # self.ki[8] = 0.008
        # self.kd[8] = -0.3
        # self.ki = np.array([0.1, 0.1, 0.1, 0.1, 0.3, 0.3, 0.3, 0.3, 0.2, 0.2, 0.2, 0.2])
        # self.kd = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        self.base = np.array([10, 10, 10, 10, -30, 67.5, 67.5, -30, -37.5, -37.5, -37.5, -37.5])
        self.halfrange = np.array([20, 20, 20, 20, 45, 67.5, 67.5, 45, 37.5, 37.5, 37.5, 37.5])
        self.errorList = []
        self.actionList = []
        self.expectList = []
        motorIndex = [10, 14, 18, 22, 8, 12, 16, 20, 0, 2, 4, 6]
        self.motorTrans = np.zeros((12, 24))
        for i in range(12):
            self.motorTrans[i][motorIndex[i]] = 1
        # self.motorDir = {"hip_z":0,"hip_y":1,"ankle":2}
        # self.legDir = {"fl":0,"fr":9, "bl":3,"br":6}

    def get_action(self, expect, state):
        actual = np.degrees(np.arcsin(self.motorTrans.dot(state[38:62])))
        actual = (actual - self.base) / self.halfrange
        # expect = -0.5
        error = expect - actual
        # self.expectList.append(actual[4])

        # print(error)
        # self.errorList.append(error[8])
        errorChange = self.old_error - error
        self.cumulative_error += error
        self.old_error = error
        action = self.kp * error + self.ki * self.cumulative_error + self.kd * errorChange
        self.actionList.append(action[0])
        return np.clip(action, -1, 1)
    # def moveforward(self,env):


