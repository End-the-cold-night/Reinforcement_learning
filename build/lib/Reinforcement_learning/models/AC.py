import numpy as np
import tensorflow as tf
OUTPUT_GRAPH = False
MAX_EPISODE = 500
DISPLAY_REWARD_THRESHOLD = 200  # renders environment if total episode reward is greater then this threshold
MAX_EP_STEPS = 2000  # maximum time step in one episode
RENDER = False  # rendering wastes time
GAMMA = 0.9  # reward discount in TD error
LR_A = 0.001  # learning rate for actor
LR_C = 0.001  # learning rate for critic
import gym
env = gym.make('CartPole-v0')
env.seed(1)  # reproducible
env = env.unwrapped

N_F = env.observation_space.shape[0]
N_A = env.action_space.n

class AC_Actor(object):
    def __init__(self, sess, n_features, n_actions, lr=0.001):
        self.sess = sess

        self.s = tf.placeholder(tf.float32, [1, n_features], "state")
        self.a = tf.placeholder(tf.int32, None, "action")
        self.q = tf.placeholder(tf.float32, None, "q")  # TD_error

        with tf.variable_scope('Actor'):
            l1 = tf.layers.dense(
                inputs=self.s,
                units=20,  # number of hidden units
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='l1'
            )

            self.acts_prob = tf.layers.dense(
                inputs=l1,
                units=n_actions,  # output units
                activation=tf.nn.softmax,  # get action probabilities
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='acts_prob'
            )

        with tf.variable_scope('exp_v'):
            log_prob = tf.log(self.acts_prob[0, self.a])
            self.exp_v = tf.reduce_mean(log_prob * self.q)  # advantage (TD_error) guided loss

        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(-self.exp_v)  # minimize(-exp_v) = maximize(exp_v)

    def learn(self, s, a, q):
        s = s[np.newaxis, :]
        feed_dict = {self.s: s, self.a: a, self.q: q}
        _, exp_v = self.sess.run([self.train_op, self.exp_v], feed_dict)
        return exp_v

    def choose_action(self, s):
        s = s[np.newaxis, :]
        probs = self.sess.run(self.acts_prob, {self.s: s})  # get probabilities for all actions
        return np.random.choice(np.arange(probs.shape[1]), p=probs.ravel())  # return a int


class AC_Critic(object):
    def __init__(self, sess, n_features,n_actions, lr=0.01):
        self.sess = sess

        self.s = tf.placeholder(tf.float32, [None, n_features], "state")
        self.a = tf.placeholder(tf.int32,[None, 1],"action")
        self.r = tf.placeholder(tf.float32, None, 'r')
        self.q_ = tf.placeholder(tf.float32,[None,1],'q_next')

        self.a_onehot = tf.one_hot(self.a, n_actions, dtype=tf.float32)
        self.a_onehot = tf.squeeze(self.a_onehot,axis=1)

        self.input = tf.concat([self.s,self.a_onehot],axis=1)

        with tf.variable_scope('Critic'):
            l1 = tf.layers.dense(
                inputs=self.input,
                units=20,  # number of hidden units
                activation=tf.nn.relu,  # None
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='l1'
            )

            self.q = tf.layers.dense(
                inputs=l1,
                units=1,  # output units
                activation=None,
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='Q'
            )

        with tf.variable_scope('squared_TD_error'):
            self.td_error = self.r + GAMMA * self.q_ - self.q
            self.loss = tf.square(self.td_error)  # TD_error = (r+gamma*V_next) - V_eval
        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss)

    def learn(self, s, a, r, s_):

        s, s_ = s[np.newaxis, :], s_[np.newaxis, :]
        # we need to find max on action, so try every action
        next_a = [[i] for i in range(N_A)]
        s_ = np.tile(s_,[N_A,1])
        q_ = self.sess.run(self.q, {self.s: s_,self.a:next_a})
        q_ = np.max(q_,axis=0,keepdims=True)
        q, _ = self.sess.run([self.q, self.train_op],
                                    {self.s: s, self.q_: q_, self.r: r,self.a:[[a]]})
        return q