# noqa
"""
# Simple

```{figure} mpe_simple.gif
:width: 140px
:name: simple
```

This environment is part of the <a href='..'>MPE environments</a>. Please read that page first for general information.

| Import             | `from pettingzoo.mpe import simple_v2` |
|--------------------|----------------------------------------|
| Actions            | Discrete/Continuous                    |
| Parallel API       | Yes                                    |
| Manual Control     | No                                     |
| Agents             | `agents= [agent_0]`                    |
| Agents             | 1                                      |
| Action Shape       | (5)                                    |
| Action Values      | Discrete(5)/Box(0.0, 1.0, (5,))        |
| Observation Shape  | (4)                                    |
| Observation Values | (-inf,inf)                             |
| State Shape        | (4,)                                   |
| State Values       | (-inf,inf)                             |


In this environment a single agent sees a landmark position and is rewarded based on how close it gets to the landmark (Euclidean distance). This is not a multiagent environment, and is primarily intended for debugging purposes.

Observation space: `[self_vel, landmark_rel_position]`

### Arguments

``` python
simple_v2.env(max_cycles=25, continuous_actions=False)
```



`max_cycles`:  number of frames (a step for each agent) until game terminates

`continuous_actions`: Whether agent action spaces are discrete(default) or continuous

"""
import sys
#sys.path.append("./multiagent-particle-envs/multiagent")
sys.path.append("/home/gong112/anaconda3/envs/doggo/lib/python3.8/site-packages/pettingzoo/mpe/")
import numpy as np
from gymnasium.utils import EzPickle
from pettingzoo.utils.conversions import parallel_wrapper_fn
# from _mpe_utils.core import Agent, Landmark, World
# from _mpe_utils.scenario import BaseScenario
# from multiagent.core import World, Agent, Landmark
# from multiagent.scenario import BaseScenario
# from _mpe_utils.simple_env import SimpleEnv, make_env

from _mpe_utils.core import Agent, Landmark, World
from _mpe_utils.scenario import BaseScenario
from _mpe_utils.simple_env import SimpleEnv, make_env

#from tianshou.utils import BaseLogger as logger

import matplotlib.pyplot as plt

class raw_env(SimpleEnv, EzPickle):
    def __init__(self, N=1, max_cycles=25, continuous_actions=False, render_mode=None):
        EzPickle.__init__(self, N, max_cycles, continuous_actions, render_mode)
        scenario = Scenario()
        world = scenario.make_world(N)
        super().__init__(
            scenario=scenario,
            world=world,
            render_mode=render_mode,
            max_cycles=max_cycles,
            continuous_actions=continuous_actions,
        )
        self.metadata["name"] = "simple_multi_v2"


env = make_env(raw_env)
parallel_env = parallel_wrapper_fn(env)


class Scenario(BaseScenario):
    def make_world(self, N=1):
        self.num_collision = 0
        self.collision = []
        world = World()
        world.dim_c = 2
        num_agents = N
        world.num_agents = num_agents
        num_adversaries = 0
        num_landmarks = max(num_agents - 1, 2)
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.adversary = True if i < num_adversaries else False
            base_name = "adversary" if agent.adversary else "agent"
            base_index = i if i < num_adversaries else i - num_adversaries
            
            agent.name = f"{base_name}_{base_index}" #f"agent_{i}"
            agent.collide = False
            agent.silent = True
            agent.size = 0.05
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = "landmark %d" % i
            landmark.collide = False
            landmark.movable = False
            landmark.size = 0.03
        return world

    def reset_world(self, world, np_random):
        # random properties for agents
        # if self.num_collision != 0:
        #     print("number of collision", self.num_collision)
        #logger.write(self, step_type = "Train", step = 0, data = {"Collision": self.num_collision})
        self.collision.append(self.num_collision)
        #print(len(self.collision) % 100)
        # if len(self.collision) % 100 == 9:
        #     print("save")
        #     plt.plot(self.collision)
            # plt.savefig("./number of collision.jpg")

        self.num_collision = 0
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.85, 0.35, 0.35])
        # random properties for landmarks
        # for i, landmark in enumerate(world.landmarks):
        #     landmark.color = np.array([0.75, 0.75, 0.75])
        # world.landmarks[0].color = np.array([0.75, 0.25, 0.25])
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.75, 0.15, 0.15])
        #print("landmarks", world.landmarks)
        goal = world.landmarks[0]
        obstacle = world.landmarks[1]
        #goal = np_random.choice(world.landmarks)
        goal.color = np.array([0.15, 0.65, 0.15])
        obstacle.color = np.array([0.5, 0.45, 0.25])
        obstacle.size = 0.05
        for agent in world.agents:
            agent.goal_a = goal
            agent.obs_a = obstacle

        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np_random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np_random.uniform(-1, +1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

    def good_agents(self, world):
        return [agent for agent in world.agents if not agent.adversary]

    # return all adversarial agents
    def adversaries(self, world):
        return [agent for agent in world.agents if agent.adversary]

    # def reward(self, agent, world):
    #     dist2 = np.sum(np.square(agent.state.p_pos - world.landmarks[0].state.p_pos))
    #     return -dist2

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
            # entity colors
        entity_color = []
        for entity in world.landmarks:
            entity_color.append(entity.color)
        #return np.concatenate([agent.state.p_vel] + entity_pos)
        # communication of all other agents
        other_pos = []
        for other in world.agents:
            if other is agent:
                continue
            other_pos.append(other.state.p_pos - agent.state.p_pos)

        if not agent.adversary:
            return np.concatenate(
                [agent.goal_a.state.p_pos - agent.state.p_pos] + entity_pos + other_pos
            )
        else:
            return np.concatenate(entity_pos + other_pos)


    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark
        return self.adversary_reward(agent, world) if agent.adversary else self.agent_reward(agent, world)

    def agent_reward(self, agent, world):
        # Rewarded based on how close any good agent is to the goal landmark, and how far the adversary is from it
        shaped_reward = True
        shaped_adv_reward = True

        # Calculate negative reward for adversary
        adversary_agents = self.adversaries(world)
        if shaped_adv_reward:  # distance-based adversary reward
            #print("adversary agent mistakenly activated")
            adv_rew = sum([np.sqrt(np.sum(np.square(a.state.p_pos - a.goal_a.state.p_pos))) for a in adversary_agents])
        else:  # proximity-based adversary reward (binary)
            adv_rew = 0
            for a in adversary_agents:
                if np.sqrt(np.sum(np.square(a.state.p_pos - a.goal_a.state.p_pos))) < 2 * a.goal_a.size:
                    adv_rew -= 5

        # Calculate positive reward for agents
        good_agents = self.good_agents(world)
        if shaped_reward:  # distance-based agent reward
            # pos_rew = -min(
            #     [np.sqrt(np.sum(np.square(a.state.p_pos - a.goal_a.state.p_pos))) for a in good_agents])
            pos_rew = -np.sqrt(np.sum(np.square(agent.state.p_pos - agent.goal_a.state.p_pos)))
                 
        else:  # proximity-based agent reward (binary)
            pos_rew = 0
            if min([np.sqrt(np.sum(np.square(a.state.p_pos - a.goal_a.state.p_pos))) for a in good_agents]) \
                    < 2 * agent.goal_a.size:
                pos_rew += 5
            pos_rew -= min(
                [np.sqrt(np.sum(np.square(a.state.p_pos - a.goal_a.state.p_pos))) for a in good_agents])

        # Calculate negative reward for obstacle collision
        obs_rew = 0
        # if min([np.sqrt(np.sum(np.square(a.state.p_pos - a.obs_a.state.p_pos))) for a in good_agents]) \
        #             < 2 * agent.obs_a.size:
        print("WHAT!!")
        if np.sqrt(np.sum(np.square(agent.state.p_pos - agent.obs_a.state.p_pos))) < 0.6 * agent.obs_a.size:
            obs_rew -= 10
            self.num_collision += 1
            print("bumped")
        return pos_rew + obs_rew #adv_rew

    def adversary_reward(self, agent, world):
        # Rewarded based on proximity to the goal landmark
        shaped_reward = True
        if shaped_reward:  # distance-based reward
            return -np.sum(np.square(agent.state.p_pos - agent.goal_a.state.p_pos))
        else:  # proximity-based reward (binary)
            adv_rew = 0
            if np.sqrt(np.sum(np.square(agent.state.p_pos - agent.goal_a.state.p_pos))) < 2 * agent.goal_a.size:
                adv_rew += 5
            return adv_rew

