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
        self.kp = np.array([1.5, 1.5, 1.5, 1.5, 3, 3, 3, 3, 3, 3, 3, 3])
        self.ki = np.array([0.1, 0.1, 0.1, 0.1, 0.3, 0.3, 0.3, 0.3, 0.2, 0.2, 0.2, 0.2])
        self.kd = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        self.base = np.array([10, 10, 10, 10, -30, 67.5, 67.5, -30, -37.5, -37.5, -37.5, -37.5])
        self.halfrange = np.array([10, 10, 10, 10, 45, 67.5, 67.5, 45, 37.5, 37.5, 37.5, 37.5])
        # self.motorDir = {"hip_z":0,"hip_y":1,"ankle":2}
        # self.legDir = {"fl":0,"fr":9, "bl":3,"br":6}

        

    # def get_actions_Motor(self, value,expect, motor):
    #     value = np.degrees(value)
    #     if motor % 3 == 0:
    #         #hip_z
    #         value = (value - 10) / 20
    #         value = np.clip(value, -1, 1)
    #         error = expect - value
    #         self.cumulative_error[motor] += error
    #         errorchange = self.old_error[motor] - error
    #         self.old_error[motor] = error
    #         action = error * self.hip_z_p
    #         action += self.cumulative_error[motor] * self.hip_z_i
    #         action += errorchange * self.hip_Z_d
    #         action = np.clip(action, -1, 1)
    #     if motor % 3 == 1:
    #         # hip_y
    #         if motor == 1 or motor == 10:
    #             value = (value + 30) / 45
    #         else:
    #             value = (value - 67.5) / 67.5
    #         value = np.clip(value, -1, 1)
    #         error = expect - value
    #         self.cumulative_error[motor] += error
    #         errorchange = self.old_error[motor] - error
    #         self.old_error[motor] = error
    #         action = error * self.hip_y_p
    #         action += self.cumulative_error[motor] * self.hip_y_i
    #         action += errorchange * self.hip_y_d
    #         action = np.clip(action, -1, 1)
    #     if motor % 3 == 2:
    #         # ankle
    #         value = (value + 37.5) / 37.5
    #         value = np.clip(value, -1, 1)
    #         error = expect - value
    #         self.cumulative_error[motor] += error
    #         errorchange = self.old_error[motor] - error
    #         self.old_error[motor] = error
    #         action = error * self.ankle_p
    #         action += self.cumulative_error[motor] * self.ankle_i
    #         action += errorchange * self.ankle_d
    #         action = np.clip(action, -1, 1)
    #
    #
    #
    #     return action
    def get_action(self,expect, actual):
        error = expect-actual
        error = (error - self.base)/self.halfrange
        errorChange = self.old_error-error
        self.cumulative_error += error
        self.old_error = error
        action = self.kp*error + self.ki* self.cumulative_error + self.kd*errorChange
        return np.clip(action, -1, 1)



if __name__ == "__main__":

    env = gym.make('Safexp-DoggoGoal2-v0')
    obs = env.reset()
    # model = PIDController()
    model = DoggoController()
    # motorIndex = [48,52,56,60,46,50,54,58,38,40,42,44]
    motorIndex = [10, 14, 18, 22, 8, 12, 16, 20, 0, 2, 4, 6]
    motorTrans = np.zeros(24,24)
    for i in range(12):
        motorTrans[i][motorIndex[i]]=1

    start_time = time.time()
    # action = model.get_actions_Forward(obs)
    test = "Forward"
    # test = "Backward"
    test = "Left"
    # test = "Right"

    for i in range(1000):

        # TODO: ADD the model to train return as targetArray
        targetArray = np.zeros(12)
        state = np.arcsin(motorTrans.dot(obs[38:62]))

        action = model.get_action(state)
        # for i in range(12):
        #     print(obs[motorIndex[i]])
        #     if obs[motorIndex[i]+1] ==0:
        #         action.append(model.get_actions_Motor(np.pi/2, targetArray[i], i))
        #     else:
        #         action.append(model.get_actions_Motor(np.arctan(obs[motorIndex[i]]/obs[motorIndex[i+1]]),targetArray[i],i))
        print(action)
        obs, reward, done, info = env.step(action)

        env.render()

        print("velocimeter", obs[101:104])
