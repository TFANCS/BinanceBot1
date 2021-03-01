from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import tensorflow as tf
import numpy as np
import pandas as pd
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
from tf_agents.networks import q_rnn_network
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common
from tf_agents.policies import policy_saver

from tf_agents.environments import suite_gym

import make_dataset
import random

tf.compat.v1.enable_v2_behavior()

num_iterations = 20000 # @param {type:"integer"}

initial_collect_steps = 1000  # @param {type:"integer"} 
collect_steps_per_iteration = 1  # @param {type:"integer"}
replay_buffer_max_length = 100000  # @param {type:"integer"}

batch_size = 64  # @param {type:"integer"}
log_interval = 250  # @param {type:"integer"}

num_eval_episodes = 1  # @param {type:"integer"}episode_return
eval_interval = 1000  # @param {type:"integer"}


time_length = 120
#time_length = 500

#self.whole_price_data : observation for all
#self.price_data : observation for one loop
#self.price_clip : observation for one step

#action
#   0:sell
#   1:wait
#   2:buy

#observation
# 5 dimension for input columns
class TradingEnv(py_environment.PyEnvironment):

    def __init__(self, price_data):
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=2, name="action")
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(12,), dtype=np.float32, minimum=0.0, name="observation")  
        self._episode_ended = False
        self.unit = 0.001
        self.time = 0
        self.whole_price_data = price_data
        self.price_data = self.whole_price_data.iloc[:time_length,:]
        self.current_price = self.price_data.iloc[self.time,0]
        self.first_quote_balance = 0.0
        self.base_balance = 0.0
        self.quote_balance = 0.0
        self.init_quote_balance = 0.0
        self.stoploss_balance = 0.0
        self.current_observation = self.price_data.iloc[self.time,1:]
        self.current_observation["BaseBalance"] = self.base_balance
        self.current_observation["QuoteBalance"] = self.quote_balance
        self.mode = 0

    def action_spec(self):
     return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def set_init_balance(self, quote_balance):
        self.init_quote_balance = quote_balance
        self.quote_balance = self.init_quote_balance

    def set_mode(self, mode):
        self.mode = mode

    def set_data(self, df):
        self.whole_price_data = df

    def _reset(self):
        self.time = 0
        random_index = random.randint(0,len(self.whole_price_data)-time_length)
        if self.mode == 0:
            self.price_data = self.whole_price_data.iloc[random_index:random_index+time_length,:]
        else:
            self.price_data = self.whole_price_data.iloc[:,:]
        self.current_price = self.price_data.iloc[self.time,0]
        self.quote_balance = self.init_quote_balance
        self.base_balance = 0.0
        self.first_quote_balance = self.quote_balance
        self._episode_ended = False
        self.current_observation = self.price_data.iloc[self.time,1:]
        self.current_observation["BaseBalance"] = self.base_balance
        self.current_observation["QuoteBalance"] = self.quote_balance
        return ts.restart(np.array(self.current_observation, dtype=np.float32))


    def _step(self, action):

        reward = 0.0

        if self._episode_ended:
            # The last action ended the episode. Ignore the current action and start
            # a new episode.
            return self.reset()

        self.time += 1

        if self.time >= len(self.price_data):
            self._episode_ended = True
        if self._episode_ended:
            reward = ((self.base_balance*self.current_price)+self.quote_balance) - self.first_quote_balance
            #print("Base_Balance:" + str(self.base_balance))
            #print("Quote_Balance:" + str(self.quote_balance))
            #print("Current Price:" + str(self.current_price))
            #print("Start_Balance:" + str(self.first_balance))
            #print("End_Balance:" + str(((self.base_balance*self.current_price)+self.quote_balance)))
            #print("Difference:" + str(reward))
            #print("")
            #reward *= 10000
            return ts.termination(np.array(self.current_observation, dtype=np.float32), reward = reward)

        self.current_price = self.price_data.iloc[self.time,0]
        self.current_observation = self.price_data.iloc[self.time,1:]
        self.current_observation["BaseBalance"] = self.base_balance
        self.current_observation["QuoteBalance"] = self.quote_balance

        # Make sure episodes don't go on forever.
        if action == 0:  #Open Sell
            self.base_balance -= self.unit
            self.quote_balance += self.unit * self.current_price * 0.999
        elif action == 2:  #buy
            self.base_balance += self.unit * 0.999
            self.quote_balance -= self.unit * self.current_price
        elif action == 1:
            pass
        else:
            raise ValueError("'action' is out of the range")


        #if self.base_balance <= 0.000001 and self.base_balance >= -0.000001: #Stop loss
        #    self.stoploss_balance = ((self.base_balance*self.current_price)+self.quote_balance)
        #if ((self.base_balance*self.current_price)+self.quote_balance) < self.stoploss_balance - (self.stoploss_balance*0.001):
        #    dif = self.base_balance
        #    self.base_balance -= dif
        #    self.quote_balance += dif * self.current_price * 0.999
        #    #print("StopLoss")

        #print(((self.base_balance*self.current_price)+self.quote_balance) - (self.stoploss_balance - (self.stoploss_balance*0.001)))
        #print(action)
        #print("Base_Balance:" + str(self.base_balance))
        #print("Quote_Balance:" + str(self.quote_balance))
        #print("Current Price:" + str(self.current_price))
        #print("All_Balance:" + str(((self.base_balance*self.current_price)+self.quote_balance)))
        #print("")
        return ts.transition(np.array(self.current_observation, dtype=np.float32), reward = reward, discount=1.0)







        
def compute_avg_return(environment, policy, num_episodes=10):

    policy_state = policy.get_initial_state(1)

    total_return = 0.0
    for _ in range(num_episodes):

        time_step = environment.reset()
        episode_return = 0.0
        while not time_step.is_last():
            action_step = policy.action(time_step, policy_state)
            #print(action_step.action.numpy()[0])
            time_step = environment.step(action_step.action)
            policy_state = action_step.state
            episode_return += time_step.reward
            #print(time_step.reward)
        #print(episode_return)
        total_return += episode_return

    avg_return = total_return / num_episodes
    return avg_return.numpy()[0]



def collect_step(environment, policy, buffer):
    time_step = environment.current_time_step()
    policy_state = policy.get_initial_state(1)
    action_step = policy.action(time_step,policy_state)
    next_time_step = environment.step(action_step.action)
    traj = trajectory.from_transition(time_step, action_step, next_time_step)

    # Add trajectory to the replay buffer
    buffer.add_batch(traj)

def collect_data(env, policy, buffer, steps):
    for _ in range(steps):
        collect_step(env, policy, buffer)






def test(binance, model):

    symbol = "BTCUSDT"

    df = pd.read_csv("..\\Data\\" + symbol + "_data.csv", index_col=0, parse_dates=True)
    #df = pd.read_csv("..\\Data\\SinSample1.csv", index_col=0, parse_dates=True)
    #df = tf.keras.utils.normalize(df, axis=0, order=2)

    df = df[["OrigClose","Close","Volume","TradeCount","BOLL_UP","BOLL_DOWN","MACD","MACD_SIGNAL","SAR","RSI12","WMA99"]]
    df = df.fillna(0)
    orig_close = df["OrigClose"]
    df = df.applymap(lambda x : 100 * x)
    df["OrigClose"] = orig_close

    train_env_py = TradingEnv(df)
    train_env_py.set_init_balance(1000)
    eval_env_py = TradingEnv(df)
    eval_env_py.set_init_balance(1000)
    #train_env_py = suite_gym.load('CartPole-v0')
    #eval_env_py = suite_gym.load('CartPole-v0')

    train_env = tf_py_environment.TFPyEnvironment(train_env_py)
    eval_env = tf_py_environment.TFPyEnvironment(eval_env_py)


    q_net = q_rnn_network.QRnnNetwork(
        train_env.observation_spec(),
        train_env.action_spec(),
        input_fc_layer_params=(128,64,16),
        output_fc_layer_params=(128,64,16),
        lstm_size=(128,64,16))

    #q_net = q_network.QNetwork(
    #    train_env.observation_spec(),
    #    train_env.action_spec(),
    #    fc_layer_params=(128,64,32))

    global_step = tf.Variable(0, name="global_step", trainable=False)


    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.001)

    agent = dqn_agent.DqnAgent(
        train_env.time_step_spec(),
        train_env.action_spec(),
        q_network=q_net,
        optimizer=optimizer,
        td_errors_loss_fn=common.element_wise_squared_loss,
        train_step_counter=global_step,
        epsilon_greedy = 0.2)

    agent.initialize()

    eval_policy = agent.policy
    collect_policy = agent.collect_policy

    random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(),
                                                    train_env.action_spec())

    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=agent.collect_data_spec,
        batch_size=train_env.batch_size,
        max_length=replay_buffer_max_length)

    collect_data(train_env, random_policy, replay_buffer, steps=360)

    policy_checkpointer = common.Checkpointer(ckpt_dir="ReinforcementLearnData/Checkpoint",
                                                agent=agent,
                                                policy=agent.policy,
                                                replay_buffer=replay_buffer,
                                                global_step=global_step)
    policy_checkpointer.initialize_or_restore()
    tf_policy_saver = policy_saver.PolicySaver(agent.policy)
    
    # Dataset generates trajectories with shape [Bx2x...]
    dataset = replay_buffer.as_dataset(
        num_parallel_calls=3, 
        sample_batch_size=batch_size, 
        num_steps=50).prefetch(3)
    
    iterator = iter(dataset)
    

    print("1:Train 2:Evaluate 3:Simulation")
    mode = input(">")
    if mode == "1":
        pass
    elif mode == "2":
        evaluate(agent, eval_env)
        return
    elif mode == "3":
        simulation(binance, agent)
        return


    # (Optional) Optimize by wrapping some of the code in a graph using TF function.
    agent.train = common.function(agent.train)
    

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
            print("step = {0}: loss = {1}".format(step, train_loss))

        if step % eval_interval == 0:
            avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
            print("step = {0}: Average Return = {1}".format(step, avg_return))
            returns.append(avg_return)



    #save agent
    policy_checkpointer.save(global_step)
    tf_policy_saver.save("ReinforcementLearnData/Policy")



    x = range(0, num_iterations + 1, eval_interval)
    plt.plot(x, returns)
    plt.ylabel('Average Return')
    plt.xlabel('Iterations')
    plt.show()

    print("Result:")
    print(compute_avg_return(eval_env, agent.policy, num_eval_episodes))




def evaluate(agent, eval_env):
    _sum = 0
    for i in range(100):
        result = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
        print(result)
        _sum += result
    print("Sum:" + str(_sum))
    print("Average:" + str(_sum/100))



def simulation(binance, agent):
    df = make_dataset.make_current_data(binance,"BTCUSDT",12,10)
    df = df[["OrigClose","Close","Volume","TradeCount","BOLL_UP","BOLL_DOWN","MACD","MACD_SIGNAL","SAR","RSI12","WMA99"]]
    df = df.fillna(0)
    orig_close = df["OrigClose"]
    df = df.applymap(lambda x : 100 * x)
    df["OrigClose"] = orig_close

    environment_py = TradingEnv(df)
    environment_py.set_init_balance(1000)
    environment_py.set_mode(1)
    environment_py.set_data(df)
    environment = tf_py_environment.TFPyEnvironment(environment_py)


    policy_state = agent.policy.get_initial_state(1)
    time_step = environment.reset()
    action_history = []
    return_sum = 0

    for _ in range(len(df)):

        action_step = agent.policy.action(time_step, policy_state)
        action_history.append(action_step.action.numpy()[0])
        time_step = environment.step(action_step.action)
        policy_state = action_step.state
        return_sum += time_step.reward

    print("Result:" + str(return_sum.numpy()[0]))


    x = range(0, len(df))
    fig = plt.figure()
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)
    ax1.plot(x, df["OrigClose"])
    ax2.plot(x, action_history)
    plt.show()

