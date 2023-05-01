
import torch
import sys
sys.path.append("/home/gong112/service_backup/work/zhaorun/doggo/safety-gym/")
import safety_gym
import gym
import numpy as np
from PPO import device, PPO_discrete
#from torch.utils.tensorboard import SummaryWriter
import os, shutil
from datetime import datetime
import sys
import pandas as pd

import argparse
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
#parser.add_argument('--write', type=str2bool, default=True, help='Use SummaryWriter to record the training')
parser.add_argument('--render', type=str2bool, default=True, help='Render or Not')
parser.add_argument('--Loadmodel', type=str2bool, default=True, help='Load pretrained model or Not')
parser.add_argument('--ModelIdex', type=int, default=10700000, help='which model to load')
# ppo_critic.pth
parser.add_argument('--seed', type=int, default=209, help='random seed')
parser.add_argument('--T_horizon', type=int, default=2048, help='lenth of long trajectory')
parser.add_argument('--Max_train_steps', type=int, default=4e7, help='Max training steps')
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

    action_model = DoggoController()
    for j in range(turns):
        s, done, ep_r, steps = env.reset(), False, 0, 0
        step_e = 0
        # for i in range(num_agent):
        #     s = s[0]
        
        while not done:
            # Take deterministic actions at test time
            step_e += 1
            action = []
            a, pi_a = model.evaluate(torch.from_numpy(s).float().to(device))
            action = action_model.get_actions(s, a)


            s_prime, r, done, info = env.step(action)

            #print("done", done[0])

            done = done
            
            # if r_flag:
            #     print("done[0][1]", done[0][1])
            #     num_collision += done[0][1]
            #print("done", done)

            r = r

            ep_r += r
            steps += 1
            s = s_prime
            #print("info", info)
            # if info['n'][0] != 0:
            #     print
            if render:
                env.render()
            #print("info", info)

        scores += ep_r
    return scores/turns

def main():
    #EnvName = ['CartPole-v1','LunarLander-v2']
    #EnvName = ['simple_multi_v2']
    #BrifEnvName = ['CP-v1','LLd-v2']
    BrifEnvName = ['Safexp-DoggoGoal2-v0']
    #env_with_Dead = [True, True]
    env_with_Dead = [True]
    EnvIdex = opt.EnvIdex
    ##env = gym.make(EnvName[EnvIdex])
    #env = make_env('simple_multi')
    env = gym.make('Safexp-DoggoGoal2-v0')
    # env.reset()
    # print("###########")
    eval_env = gym.make('Safexp-DoggoGoal2-v0')
    #eval_env = make_env('simple_multi')
    #env = simple_multi_v2.env(N=1)#, render_mode='human')
    #eval_env = simple_multi_v2.env(N=1)
    #eval_env = gym.make(EnvName[EnvIdex])
    state_dim = 104
    action_dim = 6
    max_e_steps = 1000#env._max_episode_steps

    #write = opt.write
    # if write:
    #     timenow = str(datetime.now())[0:-10]
    #     timenow = ' ' + timenow[0:13] + '_' + timenow[-2::]
    #     writepath = 'runs/{}'.format(BrifEnvName[EnvIdex]) + timenow
    #     if os.path.exists(writepath): shutil.rmtree(writepath)
    #     #writer = SummaryWriter(log_dir=writepath)

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

    env.reset()
    print("###########")

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


    #env = gym.make('Safexp-DoggoGoal2-v0')
    #obs = env.reset()
    # model = PIDController()
    action_model = DoggoController()

    
    #print("Max_train_steps", Max_train_steps)
    while total_steps < Max_train_steps:
        print("Done")
        s, done, steps, ep_r = env.reset(), False, 0, 0
        #s = s[0]
        
        #done = done[0]
        #print("done", done)
        '''Interact & trian'''
        while not done:
            traj_lenth += 1
            steps += 1
            action = []
            a_agent = []
            pi_a_agent = []
            #print("We are here!!!")
            if render:
                
                a, pi_a = model.evaluate(torch.from_numpy(s).float().to(device))  #deterministic policy
                action = action_model.get_actions(s, a)

                s_prime, r, done, info = env.step(action)
                s = s_prime
                env.render()
            else:

                a, pi_a = model.select_action(torch.from_numpy(s).float().to(device))
                action = action_model.get_actions(s, a)

                s_prime, r, done, info = env.step(action)
        
            dw = False
            model.put_data((s, a, r, s_prime, pi_a, done, dw))

            s = s_prime
            ep_r += r

            '''update if its time'''
            #print("traj_lenth!", traj_lenth)
            if not render:
                if traj_lenth % T_horizon == 0:
                    #print("training!")
                    a_loss, c_loss, entropy = model.train()
                    print("trained")
                    traj_lenth = 0
                    # if write:
                    #     writer.add_scalar('a_loss', a_loss, global_step=total_steps)
                    #     writer.add_scalar('c_loss', c_loss, global_step=total_steps)
                    #     writer.add_scalar('entropy', entropy, global_step=total_steps)

            '''record & log'''
            if total_steps % eval_interval == 0:
                #print("EVALUATING")
                score = evaluate_policy(eval_env, model, False)
                #col_period += num_collision

                # if period == 5:
                #     if col_period > 200:
                #         col_period = col_period/20
                #     collision_all.append(col_period)
                #     inter.append(int(total_steps/1000))
                #     period = 0
                #     col_period = 0
                # else:
                #     period += 1
                score_all.append(score)
                interval.append(int(total_steps/1000))
               # print("num_collision:", num_collision)
                #print("there were average" , num_collision, "collsions in the last 3 episodes")
                # if write:
                #     writer.add_scalar('ep_r', score, global_step=total_steps)
                print('EnvName:',BrifEnvName[EnvIdex],'seed:',seed,'steps: {}k'.format(int(total_steps/1000)),'score:', score)
            total_steps += 1

            #print("total_steps", total_steps)

            '''save model'''
            if total_steps % save_interval==0:
                model.save(total_steps)
                score_dict = {'steps': interval, 'scores': score_all}
                test1 = pd.DataFrame(score_dict)
                test1.to_csv("./result/doggo/scores_20_3_agent")
                plt.figure(1)
                plt.plot(interval, score_all)
                plt.title('Doggo_PID_PPO')
                plt.xlabel('total time steps')
                plt.ylabel('score')
                plt.savefig("./result/doggo/figures/scores_20_3_agent")

    score_dict = {'steps': interval, 'scores': score_all}
    test1=pd.DataFrame(score_dict)
    test1.to_csv("./result/doggo/scores_20_3_agent")
    plt.figure(1)
    plt.plot(interval, score_all)
    plt.title('Doggo_PID_PPO')
    plt.xlabel('total time steps')
    plt.ylabel('score')
    plt.savefig("./result/doggo/figures/scores_20_3_agent")

    env.close()
    eval_env.close()

if __name__ == '__main__':
    main()
