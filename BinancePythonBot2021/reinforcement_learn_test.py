from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.environments import wrappers
from tf_agents.environments import suite_gym
from tf_agents.trajectories import time_step as ts

from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import q_network
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common

tf.compat.v1.enable_v2_behavior()


num_iterations = 20000 # @param {type:"integer"}

initial_collect_steps = 1000  # @param {type:"integer"} 
collect_steps_per_iteration = 1  # @param {type:"integer"}
replay_buffer_max_length = 100000  # @param {type:"integer"}

batch_size = 64  # @param {type:"integer"}
log_interval = 200  # @param {type:"integer"}

num_eval_episodes = 10  # @param {type:"integer"}
eval_interval = 1000  # @param {type:"integer"}





#action
#  -1:sell
#   0:wait
#   1:buy

#observation
#   [base_balance,quote_balance,price]



class TradingEnv(py_environment.PyEnvironment):

    def __init__(self, price_data):
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=2, name="action")
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(3,), dtype=np.float32, minimum=0, name="observation")
        self._state = [0,0,0]
        self._episode_ended = False
        self.unit = 0.0002
        self.time = 0
        self.price_data = price_data
        self.prev_balance = 0

    def action_spec(self):
     return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self._state = [0.002, 100.0, 50000.0]
        self._episode_ended = False
        return ts.restart(np.array([self._state], dtype=np.float32))

    def _step(self, action):

        if self._episode_ended:
            # The last action ended the episode. Ignore the current action and start
            # a new episode.
            return self.reset()

        self.prev_balance = (self._state[0]*self._state[2])+self._state[1]

        price = self.price_data[self.time]

        # Make sure episodes don't go on forever.
        if action == 0:
            self._state[0] -= self.unit
            self._state[1] += self.unit * price * 0.999
        elif action == 2:
            self._state[0] += self.unit * 0.999
            self._state[1] -= self.unit * price 
        elif action == 1:
            pass
        else:
            raise ValueError("'action' is out of the range")

        self.time += 1

        if self.time > len(self.price_data):
            self._episode_ended = True

        if self._episode_ended:
            reward = self.prev_balance - ((self._state[0]*self._state[2])+self._state[1])
            return ts.termination(np.array([self._state], dtype=np.float32), reward)
        else:
            return ts.transition(np.array([self._state], dtype=np.float32), reward=0.0, discount=1.0)




        
def compute_avg_return(environment, policy, num_episodes=10):

    total_return = 0.0
    for _ in range(num_episodes):

        time_step = environment.reset()
        episode_return = 0.0

        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            episode_return += time_step.reward
        total_return += episode_return

    avg_return = total_return / num_episodes
    return avg_return.numpy()[0]



def collect_step(environment, policy, buffer):
    time_step = environment.current_time_step()
    action_step = policy.action(time_step)
    next_time_step = environment.step(action_step.action)
    traj = trajectory.from_transition(time_step, action_step, next_time_step)

    # Add trajectory to the replay buffer
    buffer.add_batch(traj)

def collect_data(env, policy, buffer, steps):
    for _ in range(steps):
        collect_step(env, policy, buffer)




def test(binance):


    df = pd.read_csv("..\\Data\\" + symbol + "_data.csv", index_col=0, parse_dates=True)
    df = df.iloc[:,1]

    environment = TradingEnv(df)
    #utils.validate_py_environment(environment, episodes=5)


    buy_action = np.array(2, dtype=np.int32)
    sell_action = np.array(0, dtype=np.int32)
    nothing_action = np.array(1, dtype=np.int32)

    time_step = environment.reset()
    print(time_step)
    cumulative_reward = time_step.reward

    for _ in range(3):
        time_step = environment.step(sell_action)
        print(time_step)
        cumulative_reward += time_step.reward

    time_step = environment.step(sell_action)
    print(time_step)
    cumulative_reward += time_step.reward
    print('Final Reward = ', cumulative_reward)



    train_env = tf_py_environment.TFPyEnvironment(CardGameEnv())
    eval_env = tf_py_environment.TFPyEnvironment(CardGameEnv())


    fc_layer_params = (100,)

    q_net = q_network.QNetwork(
        train_env.observation_spec(),
        train_env.action_spec(),
        fc_layer_params=fc_layer_params)


    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.001)

    train_step_counter = tf.Variable(0)

    agent = dqn_agent.DqnAgent(
        train_env.time_step_spec(),
        train_env.action_spec(),
        q_network=q_net,
        optimizer=optimizer,
        td_errors_loss_fn=common.element_wise_squared_loss,
        train_step_counter=train_step_counter)

    agent.initialize()

    eval_policy = agent.policy
    collect_policy = agent.collect_policy

    random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(),
                                                    train_env.action_spec())






    compute_avg_return(eval_env, random_policy, num_eval_episodes)


    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=agent.collect_data_spec,
        batch_size=train_env.batch_size,
        max_length=replay_buffer_max_length)






    collect_data(train_env, random_policy, replay_buffer, steps=100)

    # Dataset generates trajectories with shape [Bx2x...]
    dataset = replay_buffer.as_dataset(
        num_parallel_calls=3, 
        sample_batch_size=batch_size, 
        num_steps=2).prefetch(3)

    iterator = iter(dataset)


    # (Optional) Optimize by wrapping some of the code in a graph using TF function.
    agent.train = common.function(agent.train)

    # Reset the train step
    agent.train_step_counter.assign(0)


    # Evaluate the agent's policy once before training.
    avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
    returns = [avg_return]


    for _ in range(num_iterations):

        # Collect a few steps using collect_policy and save to the replay buffer.
        for _ in range(collect_steps_per_iteration):
            collect_step(train_env, agent.collect_policy, replay_buffer)

        # Sample a batch of data from the buffer and update the agent's network.
        experience, unused_info = next(iterator)
        train_loss = agent.train(experience).loss

        step = agent.train_step_counter.numpy()

        if step % log_interval == 0:
            print('step = {0}: loss = {1}'.format(step, train_loss))

        if step % eval_interval == 0:
            avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
            print('step = {0}: Average Return = {1}'.format(step, avg_return))
            returns.append(avg_return)




    x = range(0, num_iterations + 1, eval_interval)
    plt.plot(x, returns)
    plt.ylabel('Average Return')
    plt.xlabel('Iterations')
    plt.show()



