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
        self.phase = 0.0
        self.stance_phases = [0.0, 0.5, 0.5, 0.0]
        self.gait_phases = [0.0, 0.25, 0.5, 0.75]

        # PID controller constants for azimuth control
        self.azimuth_kp = 0.5
        self.azimuth_ki = 0.0
        self.azimuth_kd = 0.1
        self.azimuth_i = 0.0
        self.azimuth_last_error = 0.0

        # PID controller constants for elevation control
        self.elevation_kp = 0.5
        self.elevation_ki = 0.0
        self.elevation_kd = 0.1
        self.elevation_i = 0.0
        self.elevation_last_error = 0.0

        # PID controller constants for knee control
        self.knee_kp = 0.5
        self.knee_ki = 0.0
        self.knee_kd = 0.1
        self.knee_i = 0.0
        self.knee_last_error = 0.0

        # Default leg joint angles
        self.default_azimuth = 0.0
        self.default_elevation = 0.0
        self.default_knee = np.pi / 4

    def get_actions(self, state):
        # Get current gait and stance phases
        phase = self.phase
        stance_phases = self.stance_phases
        gait_phases = self.gait_phases

        # Update gait phase and stance phase times
        for i in range(len(stance_phases)):
            if phase < stance_phases[i]:
                stance_phase_time = stance_phases[i] - stance_phases[i-1]
                gait_phase_time = gait_phases[i] - gait_phases[i-1]
                stance_phase_completion = (phase - stance_phases[i-1]) / stance_phase_time
                gait_phase_completion = (phase - gait_phases[i-1]) / gait_phase_time
                break

        # Determine leg positions based on gait and stance phase completions
        positions = []
        for i in range(4):
            azimuth_offset = np.pi/2 * (i // 2)
            azimuth_phase = (gait_phase_completion + azimuth_offset) % 1
            elevation_phase = stance_phase_completion
            if i % 2 == 0:
                knee_phase = stance_phase_completion
            else:
                knee_phase = 1 - stance_phase_completion
            position = [azimuth_phase, elevation_phase, knee_phase]
            positions.append(position)

        # Calculate control outputs based on leg positions and current joint angles
        action = np.zeros(12)

        for leg_idx in range(4):

            # Get hip and ankle positions
            hip_y = state[46 + leg_idx*4:48 + leg_idx*4]
            hip_z = state[48 + leg_idx*4:50 + leg_idx*4]
            ankle = state[38 + leg_idx*2:40 + leg_idx*2]
            #print("hip_y", hip_y)
            #print("hip_z", hip_z)
            #print("ankle", ankle)
            #print("hip_y", np.shape(state))
            #print("sss", np.norm(hip_y-ankle))

            #Calculate leg length and projection onto the x-y plane

            # if hip_y[0] == hip_z[1]:
            #     print("hhhhhh")

            # Calculate azimuth angle (in radians)
            # current_azimuth = math.atan2(hip_y-ankle, hip_z)

            # # Calculate elevation angle (in radians)
            # current_elevation = math.atan2(proj_yz, -ankle)
            # current_knee = math.atan2(-ankle, leg_length)

            # current_azimuth = state[46 + leg_idx*4]
            # current_elevation = state[48 + leg_idx*4]
            # current_knee = state[38 + leg_idx*2]

            leg_pos = positions[leg_idx]
            azimuth_pos = leg_pos[0] * 2 * np.pi
            elevation_pos = leg_pos[1] * np.pi / 2
            knee_pos = leg_pos[2] * self.default_knee
            azimuth_error = azimuth_pos# - current_azimuth
            elevation_error = elevation_pos #- current_elevation
            knee_error = knee_pos #- current_knee
            action[leg_idx * 3 + 0] = np.clip(self.azimuth_kp * azimuth_error, -1, 1)
            action[leg_idx * 3 + 1] = np.clip(self.elevation_kp * elevation_error, -1, 1)
            action[leg_idx * 3 + 2] = np.clip(self.knee_kp * knee_error, -1, 1)


            #print("error", )
        print("action", action)
        #action = np.random.rand(12)
        return action
