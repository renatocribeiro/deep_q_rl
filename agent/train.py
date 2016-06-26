from __future__ import print_function
experiment_setup_name = "tutorial.gym.atari.seaquest-ram-v0.rnn"


#gym game title
GAME_TITLE = 'Seaquest-ram-v0'

#how many parallel game instances can your machine tolerate
N_PARALLEL_GAMES = 1

REPLAY_SEQ_LEN = 50

#theano device selection. GPU is, as always, in preference, but not required
# %env THEANO_FLAGS='device=cpu'

import gym
atari = gym.make(GAME_TITLE)
atari.reset()
n_actions = atari.action_space.n
observation_shape = (None,)+atari.observation_space.shape
del atari
RAM_SIZE = 128  # TODO:

import numpy as np
import theano
import lasagne
from lasagne.layers import InputLayer, DenseLayer
import matplotlib.pyplot as plt

obs_layer = InputLayer(observation_shape, name="observation layer")

hidden_layer_1 = lasagne.layers.DenseLayer(obs_layer,
                                           num_units=RAM_SIZE,
                                           nonlinearity=lasagne.nonlinearities.rectify,
                                           W=lasagne.init.HeUniform(),
                                           b=lasagne.init.Constant(.1))

hidden_layer_2 = lasagne.layers.DenseLayer(hidden_layer_1,
                                           num_units=RAM_SIZE,
                                           nonlinearity=lasagne.nonlinearities.rectify,
                                           W=lasagne.init.HeUniform(),
                                           b=lasagne.init.Constant(.1))

from agentnet.memory import RNNCell
hidden_state_size = 32
prev_state_layer = InputLayer([None, hidden_state_size], name='prev_state_layer')
rnn = RNNCell(prev_state_layer, hidden_layer_2, name='rnn')

q_vals = DenseLayer(hidden_layer_2, num_units=n_actions,
                    nonlinearity=lasagne.nonlinearities.linear,
                    name="QValues")

from agentnet.resolver import EpsilonGreedyResolver
start_epsilon = 0.9
end_epsilon = 0.1
epsilon = theano.shared(np.float32(start_epsilon), name="e-greedy.epsilon")
resolver = EpsilonGreedyResolver(q_vals, epsilon=epsilon, name="resolver")

from agentnet.agent import Agent
agent = Agent(obs_layer,
              agent_states={},  # ={rnn: prev_state_layer},
              policy_estimators=q_vals,
              action_layers=resolver)

# experience replay
from agentnet.environment import SessionPoolEnvironment
env = SessionPoolEnvironment(observations=obs_layer,
                             actions=resolver,
                             agent_memories=agent.agent_states)

from agentnet.experiments.openai_gym.pool import GamePool
pool = GamePool(GAME_TITLE, N_PARALLEL_GAMES)


def update_pool(env, pool, n_steps=REPLAY_SEQ_LEN):
    preceding_memory_states = list(pool.prev_memory_states)

    #get interaction sessions
    observation_tensor, action_tensor, reward_tensor, memory_logs, is_alive_tensor, _ = pool.interact(step, n_steps=n_steps)

    """print(preceding_memory_states)
    print("here:")
    print(env.preceding_agent_memories)
    print(len(env.preceding_agent_memories))
    print(len(preceding_memory_states))
    print(memory_logs)
    print(memory_logs[0].shape)"""

    #load them into experience replay environment
    print("obstensor shape: ", observation_tensor[0].shape)
    print("obs len:", len(env.observations), "obs shape: ", env.observations[0].get_value().shape)

    env.append_sessions(observation_tensor/128., action_tensor, reward_tensor,
                        is_alive_tensor, preceding_memory_states, max_pool_size=1e4)  # TODO: what does it mean?
    # TODO: make the constant global

#compile theano graph for one step decision making
applier_fun = agent.get_react_function()


def step(observation, prev_memories='zeros', batch_size=N_PARALLEL_GAMES):
    """ returns actions and new states given observation and prev state
    Prev state in default setup should be [prev window,]"""
    #default to zeros
    if prev_memories == 'zeros':
        prev_memories = [np.zeros((batch_size,)+tuple(mem.output_shape[1:]),
                         dtype='float32') for mem in agent.agent_states]
    #print("prev memories:", prev_memories)
    res = applier_fun(np.array(observation), *prev_memories)
    action = res[0]
    memories = res[1:]
#    print("new memories:", memories)
    return action, memories

# fill the pool
pool.interact(step, 1)
preceding_memory_states = list(pool.prev_memory_states)
obs_tensor, action_tensor, reward_tensor, _, is_alive_tensor, _ = pool.interact(step, REPLAY_SEQ_LEN)
env.load_sessions(obs_tensor/128., action_tensor, reward_tensor, is_alive_tensor, preceding_memory_states)


batch_size = 64
batch_env = env.sample_session_batch(batch_size)
_, _, _, _, qvalues_seq = agent.get_sessions(
    batch_env,
    session_length=REPLAY_SEQ_LEN,
    batch_size=batch_env.batch_size,
    optimize_experience_replay=True,
)

from agentnet.learning import qlearning

# gamma - delayed reward coefficient - what fraction of reward is retained if it is obtained one tick later
gamma = theano.shared(np.float32(0.95), name='q_learning_gamma')

# TODO: scale rewards
squared_Qerror = qlearning.get_elementwise_objective(
    qvalues_seq,
    batch_env.actions[0],
    batch_env.rewards,
    batch_env.is_alive,
    gamma_or_gammas=gamma)
mse_loss = squared_Qerror.sum()

from lasagne.regularization import regularize_network_params, l2
reg_l2 = regularize_network_params(resolver, l2)*10**-5
loss = mse_loss + reg_l2

weights = lasagne.layers.get_all_params(resolver, trainable=True)
updates = lasagne.updates.adadelta(loss, weights, learning_rate=0.01)

mean_session_reward = env.rewards.sum(axis=1).mean()

train_fun = theano.function([], [loss, mean_session_reward], updates=updates)
compute_mean_session_reward = theano.function([], mean_session_reward)


def display_sessions(max_n_sessions=3):
    """just draw random images"""

    plt.figure(figsize=[15, 8])

    pictures = [atari.render("rgb_array") for atari in pool.games[:max_n_sessions]]
    for i, pic in enumerate(pictures):
        plt.subplot(1, max_n_sessions, i+1)
        plt.imshow(pic)
    plt.show()

# training
from agentnet.display import Metrics
score_log = Metrics()

epoch_counter = 1
n_epochs = 25000
for i in range(n_epochs):
    #train
    update_pool(env, pool, REPLAY_SEQ_LEN)
    resolver.rng.seed(i)
    batch_env = env.sample_session_batch(batch_size)
    loss, avg_reward = train_fun()

    ##update resolver's epsilon (chance of random action instead of optimal one)
    if epoch_counter % 1 == 0:
        current_epsilon = start_epsilon + epoch_counter * ((end_epsilon - start_epsilon) / n_epochs)
        resolver.epsilon.set_value(np.float32(current_epsilon))

    ##record current learning progress and show learning curves
    if epoch_counter % 5 == 0:

        ##update learning curves
        avg_reward_current = compute_mean_session_reward()
 #       ma_reward_current = (1-alpha)*ma_reward_current + alpha*avg_reward_current
        score_log["expected e-greedy reward"][epoch_counter] = avg_reward_current

        #greedy train
        resolver.epsilon.set_value(0)
        update_pool(env, pool, REPLAY_SEQ_LEN)

        avg_reward_greedy = compute_mean_session_reward()
#        ma_reward_greedy = (1-alpha)*ma_reward_greedy + alpha*avg_reward_greedy
        score_log["expected greedy reward"][epoch_counter] = avg_reward_greedy

        #back to epsilon-greedy
        resolver.epsilon.set_value(np.float32(current_epsilon))
        update_pool(env, pool, REPLAY_SEQ_LEN)

    if epoch_counter % 100 == 0:
        print("Learning curves:")
       # score_log.plot()

        print("Random session examples")
       # display_sessions()
        print(avg_reward_greedy)

    epoch_counter += 1
