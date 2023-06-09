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
        self.ankle_p = 3
        self.ankle_i = 0.2
        self.ankle_d = 0.5
        self.hip_y_p = 3
        self.hip_y_i = 0.3
        self.hip_y_d = 0.5
        self.hip_z_p = 1.5
        self.hip_z_i = 0.1
        self.hip_Z_d = 0.5
        self.old_error = np.zeros(12)
        self.cumulative_error = np.zeros(12)
        self.motorDir = {"hip_z":0,"hip_y":1,"ankle":2}
        self.legDir = {"fl":0,"fr":9, "bl":3,"br":6}
        

    def get_actions_Motor(self, value,expect, motor):
        value = np.degrees(value)
        if motor % 3 == 0:
            #hip_z
            value = (value - 10) / 20
            value = np.clip(value, -1, 1)
            error = expect - value
            self.cumulative_error[motor] += error
            errorchange = self.old_error[motor] - error
            self.old_error[motor] = error
            action = error * self.hip_z_p
            action += self.cumulative_error[motor] * self.hip_z_i
            action += errorchange * self.hip_Z_d
            action = np.clip(action, -1, 1)
        if motor % 3 == 1:
            # hip_y
            if motor == 1 or motor == 10:
                value = (value + 30) / 45
            else:
                value = (value - 67.5) / 67.5
            value = np.clip(value, -1, 1)
            error = expect - value
            self.cumulative_error[motor] += error
            errorchange = self.old_error[motor] - error
            self.old_error[motor] = error
            action = error * self.hip_y_p
            action += self.cumulative_error[motor] * self.hip_y_i
            action += errorchange * self.hip_y_d
            action = np.clip(action, -1, 1)
        if motor % 3 == 2:
            # ankle
            value = (value + 37.5) / 37.5
            value = np.clip(value, -1, 1)
            error = expect - value
            self.cumulative_error[motor] += error
            errorchange = self.old_error[motor] - error
            self.old_error[motor] = error
            action = error * self.ankle_p
            action += self.cumulative_error[motor] * self.ankle_i
            action += errorchange * self.ankle_d
            action = np.clip(action, -1, 1)



        return action



        return action


# if __name__ == "__main__":

#     env = gym.make('Safexp-DoggoGoal2-v0')
#     obs = env.reset()
#     # model = PIDController()
#     model = DoggoController()
#     motorIndex = [48,52,56,60,46,50,54,58,38,40,42,44]

#     start_time = time.time()
#     # action = model.get_actions_Forward(obs)
#     test = "Forward"
#     # test = "Backward"
#     test = "Left"
#     # test = "Right"

#     for i in range(1000):

#         # TODO: ADD the model to train return as targetArray
#         targetArray = np.zeros(12)
#         action = []
#         for i in range(12):
#             print(obs[motorIndex[i]])
#             action.append(model.get_actions_Motor(np.arcsin(obs[motorIndex[i]]),targetArray[i],i))
#         print(action)
#         obs, reward, done, info = env.step(action)

#         #env.render()

#         print("velocimeter", obs[101:104])




import os
import glob
import time
import gym
import sys
from datetime import datetime
import safety_gym
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical

from stable_baselines3 import PPO

# #import matplotlib
# #import matplotlib.pyplot as plt

# sys.path.append("/home/gong112/service_backup/work/zhaorun/ECE593/vanilla-policy-gradient/")

# class DoggoController:
#     def __init__(self):
#         self.ankle_p = 3
#         self.ankle_i = 0.2
#         self.ankle_d = 0.5
#         self.hip_y_p = 3
#         self.hip_y_i = 0.3
#         self.hip_y_d = 0.5
#         self.hip_z_p = 1.5
#         self.hip_z_i = 0.1
#         self.hip_Z_d = 0.5
#         self.old_error = np.zeros(12)
#         self.cumulative_error = np.zeros(12)
#         self.motorDir = {"hip_z":0,"hip_y":1,"ankle":2}
#         self.legDir = {"fl":0,"fr":9, "bl":3,"br":6}
        

#     def get_actions_Motor(self, value,expect, motor):
#         error = expect-value
#         self.cumulative_error[motor] += error
#         errorchange = self.old_error[motor] -error
#         self.old_error[motor] = error
#         if motor % 3 == 0:
#             action = error * self.hip_z_p
#             action += self.cumulative_error[motor] * self.hip_z_i
#             action += errorchange * self.hip_Z_d
#             action = np.clip(action, -1, 1)
#         if motor % 3 == 1:
#             action = error * self.hip_y_p
#             action += self.cumulative_error[motor] * self.hip_y_i
#             action += errorchange * self.hip_y_d
#             action = np.clip(action, -1, 1)
#         if motor % 3 == 2:
#             action = error * self.ankle_p
#             action += self.cumulative_error[motor] * self.ankle_i
#             action += errorchange * self.ankle_d
#             action = np.clip(action, -1, 1)

#         return action


# env = gym.make('Safexp-DoggoGoal2-v0')
# # Optional: PPO2 requires a vectorized environment to run
# # the env is now wrapped automatically when passing it to the constructor
# # env = DummyVecEnv([lambda: env])

# model = PPO("MlpPolicy", env, verbose=1)
# model.learn(total_timesteps=10000)

# action_model = DoggoController()
# motorIndex = [48,52,56,60,46,50,54,58,38,40,42,44]

# #vec_env = model.get_env()
# obs = env.reset()

# #obs = env.reset()
# for i in range(1000):
#     motor_control = []
#     action, _states = model.predict(obs)
#     # for i in range(12):
#     #     motor_control.append(model.get_actions_Motor(np.arcsin(obs[motorIndex[i]]),action[i],i))

#     #print("motor control", motor_control)
#     #obs, rewards, dones, info = vec_env.step(motor_control)
#     obs, rewards, dones, info = env.step(action)
#     #env.render()

# env.close()



# set device to cpu or cuda
device = torch.device('cpu')
if(torch.cuda.is_available()): 
    device = torch.device('cuda:0') 
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")

class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []
    
    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, has_continuous_action_space, action_std_init):
        super(ActorCritic, self).__init__()

        self.has_continuous_action_space = has_continuous_action_space
        
        if has_continuous_action_space:
            self.action_dim = action_dim
            self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)
        # actor
        if has_continuous_action_space :
            self.actor = nn.Sequential(
                            nn.Linear(state_dim, 64),
                            nn.Tanh(),
                            nn.Linear(64, 64),
                            nn.Tanh(),
                            nn.Linear(64, action_dim),
                            nn.Tanh()
                        )
        else:
            self.actor = nn.Sequential(
                            nn.Linear(state_dim, 64),
                            nn.Tanh(),
                            nn.Linear(64, 64),
                            nn.Tanh(),
                            nn.Linear(64, action_dim),
                            nn.Softmax(dim=-1)
                        )
        # critic
        self.critic = nn.Sequential(
                        nn.Linear(state_dim, 64),
                        nn.Tanh(),
                        nn.Linear(64, 64),
                        nn.Tanh(),
                        nn.Linear(64, 1)
                    )
        
    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(device)

    def forward(self):
        raise NotImplementedError
    
    def act(self, state):

        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
            dist = MultivariateNormal(action_mean, cov_mat)
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)
        state_val = self.critic(state)

        return action.detach(), action_logprob.detach(), state_val.detach()
    
    def evaluate(self, state, action):

        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            
            action_var = self.action_var.expand_as(action_mean)
            cov_mat = torch.diag_embed(action_var).to(device)
            dist = MultivariateNormal(action_mean, cov_mat)
            
            # For Single Action Environments.
            if self.action_dim == 1:
                action = action.reshape(-1, self.action_dim)
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)
        
        return action_logprobs, state_values, dist_entropy


class PPO:
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std_init=0.6):

        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_std = action_std_init

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
        self.optimizer = torch.optim.Adam([
                        {'params': self.policy.actor.parameters(), 'lr': lr_actor},
                        {'params': self.policy.critic.parameters(), 'lr': lr_critic}
                    ])

        self.policy_old = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()

    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_std = new_action_std
            self.policy.set_action_std(new_action_std)
            self.policy_old.set_action_std(new_action_std)

    def decay_action_std(self, action_std_decay_rate, min_action_std):
        if self.has_continuous_action_space:
            self.action_std = self.action_std - action_std_decay_rate
            self.action_std = round(self.action_std, 4)
            if (self.action_std <= min_action_std):
                self.action_std = min_action_std
            self.set_action_std(self.action_std)

    def select_action(self, state):

        if self.has_continuous_action_space:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(device)
                action, action_logprob, state_val = self.policy_old.act(state)

            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)
            self.buffer.state_values.append(state_val)

            return action.detach().cpu().numpy().flatten()
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(device)
                action, action_logprob, state_val = self.policy_old.act(state)
            
            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)
            self.buffer.state_values.append(state_val)

            return action.item()

    def update(self):
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach().to(device)

        # calculate advantages
        advantages = rewards.detach() - old_state_values.detach()

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):

            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)
            
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss  
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()
    
    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)
   
    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        

def train():


    has_continuous_action_space = True  # continuous action space; else discrete

    max_ep_len = 1000                   # max timesteps in one episode
    max_training_timesteps = int(3e6)   # break training loop if timeteps > max_training_timesteps
    #max_training_timesteps = int(3e5)
    print_freq = max_ep_len * 10        # print avg reward in the interval (in num timesteps)
    log_freq = max_ep_len * 2           # log avg reward in the interval (in num timesteps)
    save_model_freq = int(1e5)          # save model frequency (in num timesteps)

    action_std = 0.6                    # starting std for action distribution (Multivariate Normal)
    action_std_decay_rate = 0.05        # linearly decay action_std (action_std = action_std - action_std_decay_rate)
    min_action_std = 0.1                # minimum action_std (stop decay after action_std <= min_action_std)
    action_std_decay_freq = int(2.5e5)  # action_std decay frequency (in num timesteps)
    update_timestep = max_ep_len * 4      # update policy every n timesteps
    K_epochs = 80               # update policy for K epochs in one PPO update

    eps_clip = 0.2          # clip parameter for PPO
    gamma = 0.9            # discount factor

    lr_actor = 0.0003       # learning rate for actor network
    lr_critic = 0.001       # learning rate for critic network

    random_seed = 0         # set random seed if required (0 = no random seed)
    env_name = "Reacher"
    #env = gym.make("modified_gym_env:ReacherPyBulletEnv-v1", rand_init=True)
    env = gym.make('Safexp-DoggoGoal2-v0')

    # state space dimension
    state_dim = 104#

    # action space dimension
    if has_continuous_action_space:
        action_dim = 12
    else:
        action_dim = env.action_space.n


    #### log files for multiple runs are NOT overwritten
    log_dir = "PPO_logs"
    if not os.path.exists(log_dir):
          os.makedirs(log_dir)

    log_dir = log_dir + '/' + env_name + '/'
    if not os.path.exists(log_dir):
          os.makedirs(log_dir)

    #### get number of log files in log directory
    run_num = 0
    current_num_files = next(os.walk(log_dir))[2]
    run_num = len(current_num_files)

    #### create new log file for each run
    log_f_name = log_dir + '/PPO_' + env_name + "_log_" + str(run_num) + ".csv"

    print("current logging run number for " + env_name + " : ", run_num)
    print("logging at : " + log_f_name)

    ################### checkpointing ###################
    run_num_pretrained = 0     
    directory = "PPO_preTrained"
    if not os.path.exists(directory):
          os.makedirs(directory)

    directory = directory + '/' + env_name + '/'
    if not os.path.exists(directory):
          os.makedirs(directory)


    checkpoint_path = directory + "PPO_{}_{}_{}.pth".format(env_name, random_seed, run_num_pretrained)
    print("save checkpoint path : " + checkpoint_path)

    if random_seed:
        print("--------------------------------------------------------------------------------------------")
        print("setting random seed to ", random_seed)
        torch.manual_seed(random_seed)
        env.seed(random_seed)
        np.random.seed(random_seed)

    # initialize a PPO agent
    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std)

    # track total training time
    start_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)

    # logging file
    log_f = open(log_f_name,"w+")
    log_f.write('episode,timestep,reward\n')

    # printing and logging variables
    print_running_reward = 0
    print_running_episodes = 0

    log_running_reward = 0
    log_running_episodes = 0

    time_step = 0
    i_episode = 0

    log_avg_rewards = []

    render = False

    # training loop
    while time_step <= max_training_timesteps:
        if render:
            env.render()

        state = env.reset()
        current_ep_reward = 0

        action_model = DoggoController()
        motorIndex = [48,52,56,60,46,50,54,58,38,40,42,44]

        for t in range(1, max_ep_len+1):

            if render:
                env.render()
            # select action with policy
            action = ppo_agent.select_action(state)
            targetArray = action #np.zeros(12)
            action = []
            obs = state
            for i in range(12):
                # print(state[motorIndex[i]])
                action.append(action_model.get_actions_Motor(np.arcsin(obs[motorIndex[i]]),targetArray[i],i))
            # print(action)
            #obs, reward, done, info = env.step(action)




            state, reward, done, _ = env.step(action)

            # saving reward and is_terminals
            ppo_agent.buffer.rewards.append(reward)
            ppo_agent.buffer.is_terminals.append(done)

            time_step +=1
            current_ep_reward += reward

            # update PPO agent
            if time_step % update_timestep == 0:
                ppo_agent.update()

            # if continuous action space; then decay action std of ouput action distribution
            if has_continuous_action_space and time_step % action_std_decay_freq == 0:
                ppo_agent.decay_action_std(action_std_decay_rate, min_action_std)

            # log in logging file
            if time_step % log_freq == 0:

                # log average reward till last episode
                log_avg_reward = log_running_reward / log_running_episodes
                log_avg_reward = round(log_avg_reward, 4)

                log_f.write('{},{},{}\n'.format(i_episode, time_step, log_avg_reward))
                log_f.flush()

                log_running_reward = 0
                log_running_episodes = 0

            # printing average reward
            if time_step % print_freq == 0:

                # print average reward till last episode
                print_avg_reward = print_running_reward / print_running_episodes
                print_avg_reward = round(print_avg_reward, 2)

                print("Episode : {} \t\t Timestep : {} \t\t Average Reward : {}".format(i_episode, time_step, print_avg_reward))
                log_avg_rewards.append(print_avg_reward)
                print_running_reward = 0
                print_running_episodes = 0

            # save model weights
            if time_step % save_model_freq == 0:
                print("saving model at : " + checkpoint_path)
                ppo_agent.save(checkpoint_path)
                print("model saved")
                print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)

            # break; if the episode is over
            if done:
                break

        print_running_reward += current_ep_reward
        print_running_episodes += 1

        log_running_reward += current_ep_reward
        log_running_episodes += 1

        i_episode += 1

    log_f.close()
    env.close()

    # plt.scatter(np.arange(len(log_avg_rewards)), log_avg_rewards, label='individual iterations')
    # plt.plot(log_avg_rewards, label="average reward of iteration")
    # plt.title(f'Average Reward at Each Iteration')
    # plt.xlabel('Episode')
    # plt.ylabel('Reward')
    # plt.legend()
    # plt.savefig('Q2_PPO: Average Reward at Each Iteration')

    # print total training time
    end_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print("Finished training at (GMT) : ", end_time)
    print("Total training time  : ", end_time - start_time)
 

if __name__ == '__main__':

    train()
    
