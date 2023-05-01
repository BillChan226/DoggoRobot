import gym
import torch
import numpy as np
from PPO import device, PPO_discrete
from torch.utils.tensorboard import SummaryWriter
import os, shutil
from datetime import datetime
import sys
import pandas as pd
#sys.path.append("./multiagent-particle-envs/multiagent")
sys.path.append("/home/gong112/service_backup/work/zhaorun/multi_CAVs/")

import argparse
from MPE.make_env import make_env
#import simple_multi_v2
import matplotlib.pyplot as plt
from doggo_PID import DoggoController



def str2bool(v):
    '''transfer str to bool for argparse'''
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'True','true','TRUE', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'False','false','FALSE', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

'''Hyperparameter Setting'''
parser = argparse.ArgumentParser()
parser.add_argument('--EnvIdex', type=int, default=0, help='CP-v1, LLd-v2')
parser.add_argument('--write', type=str2bool, default=True, help='Use SummaryWriter to record the training')
parser.add_argument('--render', type=str2bool, default=True, help='Render or Not')
parser.add_argument('--Loadmodel', type=str2bool, default=False, help='Load pretrained model or Not')
parser.add_argument('--ModelIdex', type=int, default=300000, help='which model to load')

parser.add_argument('--seed', type=int, default=209, help='random seed')
parser.add_argument('--T_horizon', type=int, default=2048, help='lenth of long trajectory')
parser.add_argument('--Max_train_steps', type=int, default=4e5, help='Max training steps')
parser.add_argument('--save_interval', type=int, default=1e5, help='Model saving interval, in steps.')
parser.add_argument('--eval_interval', type=int, default=5e3, help='Model evaluating interval, in steps.')

parser.add_argument('--gamma', type=float, default=0.99, help='Discounted Factor')
parser.add_argument('--lambd', type=float, default=0.95, help='GAE Factor')
parser.add_argument('--clip_rate', type=float, default=0.2, help='PPO Clip rate')
parser.add_argument('--K_epochs', type=int, default=10, help='PPO update times')
parser.add_argument('--net_width', type=int, default=64, help='Hidden net width')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--l2_reg', type=float, default=0, help='L2 regulization coefficient for Critic')
parser.add_argument('--batch_size', type=int, default=64, help='lenth of sliced trajectory')
parser.add_argument('--entropy_coef', type=float, default=0, help='Entropy coefficient of Actor')
parser.add_argument('--entropy_coef_decay', type=float, default=0.99, help='Decay rate of entropy_coef')
parser.add_argument('--adv_normalization', type=str2bool, default=False, help='Advantage normalization')
opt = parser.parse_args()
print(opt)


def evaluate_policy(env, model, render):
    scores = 0
    turns = 3
    num_collision = 0
    num_agent = 8
    for j in range(turns):
        s, done, ep_r, steps = env.reset(), False, 0, 0
        step_e = 0
        # for i in range(num_agent):
        #     s = s[0]
        
        while not done:
            # Take deterministic actions at test time
            step_e += 1
            action = []
            for i in range(num_agent):
                a, pi_a = model.evaluate(torch.from_numpy(s[i]).float().to(device))

                # if a == 0: action=[[1,0,0,0,0]]
                # elif a == 1: action=[[0,1,0,0,0]]
                # elif a == 2: action=[[0,0,1,0,0]]
                # elif a == 3: action=[[0,0,0,1,0]]
                # elif a == 4: action=[[0,0,0,0,1]]

                if a == 0: action.append([1,0,0,0,0])
                elif a == 1: action.append([0,1,0,0,0])
                elif a == 2: action.append([0,0,1,0,0])
                elif a == 3: action.append([0,0,0,1,0])
                elif a == 4: action.append([0,0,0,0,1])

            s_prime, r, done, info = env.step(action)

            #print("done", done[0])

            done = done[0]
            
            # if r_flag:
            #     print("done[0][1]", done[0][1])
            #     num_collision += done[0][1]
            #print("done", done)

            r = r[0]

            ep_r += r
            steps += 1
            s = s_prime
            #print("info", info)
            # if info['n'][0] != 0:
            #     print
            if done:
                #print("DONE")
                num_collision += info['n'][0]
                #print("lalala:", num_collision)
            #s = s[0]
            if render:
                env.render()
            #print("info", info)
            if step_e >= 1500:
                num_collision += info['n'][0]
                break

        scores += ep_r
    return scores/turns, num_collision

def main():
    #EnvName = ['CartPole-v1','LunarLander-v2']
    #EnvName = ['simple_multi_v2']
    #BrifEnvName = ['CP-v1','LLd-v2']
    BrifEnvName = ['SM-v2']
    #env_with_Dead = [True, True]
    env_with_Dead = [True]
    EnvIdex = opt.EnvIdex
    ##env = gym.make(EnvName[EnvIdex])
    env = make_env('simple_multi')
    eval_env = make_env('simple_multi')
    #env = simple_multi_v2.env(N=1)#, render_mode='human')
    #eval_env = simple_multi_v2.env(N=1)
    #eval_env = gym.make(EnvName[EnvIdex])
    state_dim = env.observation_space[0].shape[0]
    action_dim = env.action_space[0].n
    max_e_steps = 1000#env._max_episode_steps

    write = opt.write
    if write:
        timenow = str(datetime.now())[0:-10]
        timenow = ' ' + timenow[0:13] + '_' + timenow[-2::]
        writepath = 'runs/{}'.format(BrifEnvName[EnvIdex]) + timenow
        if os.path.exists(writepath): shutil.rmtree(writepath)
        writer = SummaryWriter(log_dir=writepath)

    T_horizon = opt.T_horizon
    render = opt.render
    Loadmodel = opt.Loadmodel
    ModelIdex = opt.ModelIdex #which model to load
    Max_train_steps = opt.Max_train_steps #in steps
    eval_interval = opt.eval_interval #in steps
    save_interval = opt.save_interval #in steps

    seed = opt.seed
    torch.manual_seed(seed)
    env.seed(seed)
    eval_env.seed(seed)
    np.random.seed(seed)

    print('Env:',BrifEnvName[EnvIdex],'  state_dim:',state_dim,'  action_dim:',action_dim,'   Random Seed:',seed, '  max_e_steps:',max_e_steps)
    print('\n')

    kwargs = {
        "env_with_Dead": env_with_Dead[EnvIdex],
        "state_dim": state_dim,
        "action_dim": action_dim,
        "gamma": opt.gamma,
        "lambd": opt.lambd,
        "net_width": opt.net_width,
        "lr": opt.lr,
        "clip_rate": opt.clip_rate,
        "K_epochs": opt.K_epochs,
        "batch_size": opt.batch_size,
        "l2_reg":opt.l2_reg,
        "entropy_coef":opt.entropy_coef,  #hard env needs large value
        "adv_normalization":opt.adv_normalization,
        "entropy_coef_decay": opt.entropy_coef_decay,
    }

    if not os.path.exists('model'): os.mkdir('model')
    model = PPO_discrete(**kwargs)
    if Loadmodel: model.load(ModelIdex)


    traj_lenth = 0
    total_steps = 0
    score_all = []
    interval = []
    collision_all = []
    #num_collision = []
    col_period = 0
    inter = []
    period = 0
    num_agent = 8

    while total_steps < Max_train_steps:
        s, done, steps, ep_r = env.reset(), False, 0, 0
        #s = s[0]

        #done = done[0]
        '''Interact & trian'''
        while not done:
            traj_lenth += 1
            steps += 1
            action = []
            a_agent = []
            pi_a_agent = []
            if render:
                for i in range(num_agent):
                # a, pi_a = model.select_action(torch.from_numpy(s).float().to(device))  #stochastic policy
                    a, pi_a = model.evaluate(torch.from_numpy(s[i]).float().to(device))  #deterministic policy
                    a_agent.append(a)
                    pi_a_agent.append(pi_a)
                    if a == 0: action.append([1,0,0,0,0])
                    elif a == 1: action.append([0,1,0,0,0])
                    elif a == 2: action.append([0,0,1,0,0])
                    elif a == 3: action.append([0,0,0,1,0])
                    elif a == 4: action.append([0,0,0,0,1])

                s_prime, r, done, info = env.step(action)
                s = s_prime
                env.render()
            else:
                for i in range(num_agent):
                    a, pi_a = model.select_action(torch.from_numpy(s[i]).float().to(device))
                    a_agent.append(a)
                    pi_a_agent.append(pi_a)
                    if a == 0: action.append([1,0,0,0,0])
                    elif a == 1: action.append([0,1,0,0,0])
                    elif a == 2: action.append([0,0,1,0,0])
                    elif a == 3: action.append([0,0,0,1,0])
                    elif a == 4: action.append([0,0,0,0,1])

            s_prime, r, done, info = env.step(action)
            # print("action", action)
            # print("r", r)
            # print("s_prime", s_prime)
            #print("info", info)

            #print(done)

            #s_prime = s_prime[0]

            #r = r[0]

            #done = done[0]
            # if done == True:
            #     print("steps", steps)
            # if done:
            #     num_collision.append(info['n'][0])
            
            # if (done and steps != max_e_steps):
            #     if EnvIdex == 1:
            #         if r <=-100: r = -30  #good for LunarLander
            #     dw = True  #dw: dead and win
            # else:
            #     dw = False
            dw = False
            for i in range(num_agent):
                model.put_data((s[i], a_agent[i], r[i], s_prime[i], pi_a_agent[i], done[i], dw))

            s = s_prime
            ep_r += r[0]

            '''update if its time'''
            #print("traj_lenth!", traj_lenth)
            if not render:
                if traj_lenth % T_horizon == 0:
                    #print("training!")
                    a_loss, c_loss, entropy = model.train()
                    traj_lenth = 0
                    if write:
                        writer.add_scalar('a_loss', a_loss, global_step=total_steps)
                        writer.add_scalar('c_loss', c_loss, global_step=total_steps)
                        writer.add_scalar('entropy', entropy, global_step=total_steps)

            '''record & log'''
            if total_steps % eval_interval == 0:
                #print("EVALUATING")
                score, num_collision = evaluate_policy(eval_env, model, False)
                col_period += num_collision

                if period == 5:
                    if col_period > 200:
                        col_period = col_period/20
                    collision_all.append(col_period)
                    inter.append(int(total_steps/1000))
                    period = 0
                    col_period = 0
                else:
                    period += 1
                score_all.append(score)
                interval.append(int(total_steps/1000))
                print("num_collision:", num_collision)
                #print("there were average" , num_collision, "collsions in the last 3 episodes")
                if write:
                    writer.add_scalar('ep_r', score, global_step=total_steps)
                print('EnvName:',BrifEnvName[EnvIdex],'seed:',seed,'steps: {}k'.format(int(total_steps/1000)),'score:', score)
            total_steps += 1

            #print("total_steps", total_steps)

            '''save model'''
            # if total_steps % save_interval==0:
            #     model.save(total_steps)
    score_dict = {'steps': interval, 'scores': score_all}
    test1=pd.DataFrame(score_dict)
    test1.to_csv("./result/multiagents/scores_20_3_agent")
    plt.figure(1)
    plt.plot(interval, score_all)
    plt.title('Averaged Scores of 0.2 Observability with 3 Agents')
    plt.xlabel('total time steps')
    plt.ylabel('score')
    plt.savefig("./result/multiagents/figures/scores_20_3_agent")

    col_dict = {'steps': inter, 'col': collision_all}
    test2=pd.DataFrame(col_dict)
    test2.to_csv("./result/multiagents/col_20_3_agent")

    plt.figure(2)
    plt.plot(inter, collision_all)
    plt.title('Averaged Collision of 0.2 Observability with 3 Agents')
    plt.xlabel('total time steps')
    plt.ylabel('number of collisions')
    plt.savefig("./result/multiagents/figures/col_20_3_agent")

    env.close()
    eval_env.close()

if __name__ == '__main__':
    main()
