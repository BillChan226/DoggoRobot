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


# class DoggoController:
#     def __init__(self):
#         self.phase = 0.0
#         self.stance_phases = [0.0, 0.5, 0.5, 0.0]
#         self.gait_phases = [0.0, 0.25, 0.5, 0.75]

#         # PID controller constants for azimuth control
#         self.azimuth_kp = 0.5
#         self.azimuth_ki = 0.0
#         self.azimuth_kd = 0.1
#         self.azimuth_i = 0.0
#         self.azimuth_last_error = 0.0

#         # PID controller constants for elevation control
#         self.elevation_kp = 0.5
#         self.elevation_ki = 0.0
#         self.elevation_kd = 0.1
#         self.elevation_i = 0.0
#         self.elevation_last_error = 0.0

#         # PID controller constants for knee control
#         self.knee_kp = 0.5
#         self.knee_ki = 0.0
#         self.knee_kd = 0.1
#         self.knee_i = 0.0
#         self.knee_last_error = 0.0

#         # Default leg joint angles
#         self.default_azimuth = 0.0
#         self.default_elevation = 0.0
#         self.default_knee = np.pi / 4

#     def get_actions(self, state):
#         # Get current gait and stance phases
#         phase = self.phase
#         stance_phases = self.stance_phases
#         gait_phases = self.gait_phases

#         # Update gait phase and stance phase times
#         for i in range(len(stance_phases)):
#             if phase < stance_phases[i]:
#                 stance_phase_time = stance_phases[i] - stance_phases[i-1]
#                 gait_phase_time = gait_phases[i] - gait_phases[i-1]
#                 stance_phase_completion = (phase - stance_phases[i-1]) / stance_phase_time
#                 gait_phase_completion = (phase - gait_phases[i-1]) / gait_phase_time
#                 break

#         # Determine leg positions based on gait and stance phase completions
#         positions = []
#         for i in range(4):
#             azimuth_offset = np.pi/2 * (i // 2)
#             azimuth_phase = (gait_phase_completion + azimuth_offset) % 1
#             elevation_phase = stance_phase_completion
#             if i % 2 == 0:
#                 knee_phase = stance_phase_completion
#             else:
#                 knee_phase = 1 - stance_phase_completion
#             position = [azimuth_phase, elevation_phase, knee_phase]
#             positions.append(position)

#         # Calculate control outputs based on leg positions and current joint angles
#         action = np.zeros(12)

#         for leg_idx in range(4):

#             # Get hip and ankle positions
#             hip_y = state[46 + leg_idx*4:48 + leg_idx*4]
#             hip_z = state[48 + leg_idx*4:50 + leg_idx*4]
#             ankle = state[38 + leg_idx*2:40 + leg_idx*2]
#             #print("hip_y", hip_y)
#             #print("hip_z", hip_z)
#             #print("ankle", ankle)
#             #print("hip_y", np.shape(state))
#             #print("sss", np.norm(hip_y-ankle))

#             #Calculate leg length and projection onto the x-y plane

#             # if hip_y[0] == hip_z[1]:
#             #     print("hhhhhh")

#             # Calculate azimuth angle (in radians)
#             # current_azimuth = math.atan2(hip_y-ankle, hip_z)

#             # # Calculate elevation angle (in radians)
#             # current_elevation = math.atan2(proj_yz, -ankle)
#             # current_knee = math.atan2(-ankle, leg_length)

#             # current_azimuth = state[46 + leg_idx*4]
#             # current_elevation = state[48 + leg_idx*4]
#             # current_knee = state[38 + leg_idx*2]

#             leg_pos = positions[leg_idx]
#             azimuth_pos = leg_pos[0] * 2 * np.pi
#             elevation_pos = leg_pos[1] * np.pi / 2
#             knee_pos = leg_pos[2] * self.default_knee
#             azimuth_error = azimuth_pos# - current_azimuth
#             elevation_error = elevation_pos #- current_elevation
#             knee_error = knee_pos #- current_knee
#             action[leg_idx * 3 + 0] = np.clip(self.azimuth_kp * azimuth_error, -1, 1)
#             action[leg_idx * 3 + 1] = np.clip(self.elevation_kp * elevation_error, -1, 1)
#             action[leg_idx * 3 + 2] = np.clip(self.knee_kp * knee_error, -1, 1)


#             #print("error", )
#         print("action", action)
#         #action = np.random.rand(12)
#         return action


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
        self.ki = 1
        self.kd = 0.5
        self.dirzOld = np.zeros(4)
        self.dirzSum = np.zeros(4)

    def get_actions_Forward(self, observation):

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
        dir_z = np.zeros(4)
        for i in range(4):
            dir_z[i] = observation[48 + 4 * i] - hip_z_phase[i]
        self.dirzSum = np.add(dir_z, self.dirzSum)
        for i in range(4):
            action[i] = -np.clip(dir_z[i] * self.kp + self.dirzSum[i] *self.ki + (self.dirzOld[i]-dir_z[i])*self.kd, -1, 1)

            dir_y = observation[46 + 4 * i]  # + self.hip_y_phase[i]

            action[i + 4] = -np.clip(dir_y, -1, 1)

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
    action = model.get_actions_Forward(obs)

    for i in range(1000):
        # print("obs", obs)
        # for sensor in env.sensors_obs:  # Explicitly listed sensors
        #         dim = env.robot.sensor_dim[sensor]
        #         print("obs-dim",sensor, dim)
        # action = np.ones(12)
        # print("action", action)
        action = model.get_actions_Forward(obs)

        obs, reward, done, info = env.step(action)

        env.render()

        print("velocimeter", obs[101:104])
