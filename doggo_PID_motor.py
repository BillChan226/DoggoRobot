# from safety_gym import safety_gym
import sys

sys.path.append("/home/gong112/service_backup/work/zhaorun/doggo/safety-gym/")
import safety_gym
import gym
import gym.envs.registration as r
import os
import math
import time
import matplotlib.pyplot as plt
# print(r)

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
        self.kp = np.array([0.4, 0.3, 0.3, 0.4, 0.8, 0.8, 0.3, 0.3, 1.8, 1.8, 0.4, 0.4])
        # self.ki = np.array([0.0005, 0.0005, 0.0005, 0.0005, 0.015, 0.015, 0.015, 0.015, 0.008, 0.008, 0.008, 0.008])
        # self.kd = np.array([-0.06, -0.06, -0.06, -0.06, -0.15, -0.15, -0.15, -0.15, -0.3, -0.3, -0.3, -0.3])
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
        for i in range(12):
            self.errorList.append([])
        self.actionList=[]
        self.expectList = []
        motorIndex = [10, 14, 18, 22, 8, 12, 16, 20, 0, 2, 4, 6]
        self.motorTrans = np.zeros((12, 24))
        for i in range(12):
            self.motorTrans[i][motorIndex[i]] = 1
        # self.motorDir = {"hip_z":0,"hip_y":1,"ankle":2}
        # self.legDir = {"fl":0,"fr":9, "bl":3,"br":6}

        

    def get_action(self,expect, state):
        actual = np.degrees(np.arcsin(self.motorTrans.dot(state[38:62])))
        actual = (actual - self.base) / self.halfrange
        # expect = -0.5
        error = expect-actual
        self.expectList.append(actual[4])

        # print(error)
        for i in range(12):
            self.errorList[i].append(error[i])
        errorChange = self.old_error-error
        self.cumulative_error += error
        self.old_error = error
        action = self.kp*error + self.ki* self.cumulative_error + self.kd*errorChange
        self.actionList.append(action[0])
        return np.clip(action, -1, 1)
    def moveforward(self,env):
        #move left
        targetArray = np.zeros(12)






if __name__ == "__main__":

    env = gym.make('Safexp-DoggoGoal2-v0')
    obs = env.reset()
    # model = PIDController()
    model = DoggoController()
    # motorIndex = [48,52,56,60,46,50,54,58,38,40,42,44]
    # motorIndex = [10, 14, 18, 22, 8, 12, 16, 20, 0, 2, 4, 6]
    # motorTrans = np.zeros((12,24))
    # for i in range(12):
    #     motorTrans[i][motorIndex[i]]=1

    start_time = time.time()
    # action = model.get_actions_Forward(obs)
    test = "Forward"
    # test = "Backward"
    test = "Left"
    # test = "Right"

    for i in range(200):

        # TODO: ADD the model to train return as targetArray
        targetArray = np.zeros(12)
        # state = np.degrees(np.arcsin(motorTrans.dot(obs[38:62])))
        # state = motorTrans.dot(state)
        action = model.get_action(targetArray,obs)
        # for i in range(12):
        #     print(obs[motorIndex[i]])
        #     if obs[motorIndex[i]+1] ==0:
        #         action.append(model.get_actions_Motor(np.pi/2, targetArray[i], i))
        #     else:
        #         action.append(model.get_actions_Motor(np.arctan(obs[motorIndex[i]]/obs[motorIndex[i+1]]),targetArray[i],i))
        # print(action)
        obs, reward, done, info = env.step(action)

        env.render()

        # print("velocimeter", obs[101:104])
    plt.figure(0)
    for i in range(12):
        plt.figure(i%4)
        plt.plot(model.errorList[i],label = str(int(i)))
        plt.legend()
    # plt.figure(1)
    # plt.plot(model.actionList)
    # plt.figure(2)
    # plt.plot(model.expectList)
    plt.legend()
    plt.show()