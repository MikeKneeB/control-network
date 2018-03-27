"""
Data structure for implementing critic network for DDPG algorithm
Algorithm and hyperparameter details can be found here:
    http://arxiv.org/pdf/1509.02971v2.pdf

Original author: Patrick Emami

Author: Bart Keulen

Further comments added by Mike Knee.
"""

import tensorflow as tf
import tflearn


class CriticNetwork(object):

    """
    Constructor for the CriticNetwork object.

    sess: tensorflow session, necessary to run all tensorflow operations.
    state_dim: dimensions of the state space of the environment this object
        will run in.
    action_dim: dimensions of the action space of the environment this object
        will run in.
    action_bound: upper limit of the action, assumed action can take any value
        between -action_bound and +action_bound.
    learning_rate: rate defining how quickly the agent will learn. 0 -> 1
    tau: rate defining how quickly the target network changes. 0 -> 1
    num_actor_vars: number of trainable variables in the actor network. This
        should be generated by the accompanying ActorNetwork.
    """
    def __init__(self, sess, state_dim, action_dim, action_bound, learning_rate, tau, num_actor_vars):
        self.sess = sess
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound
        self.learning_rate = learning_rate
        self.tau = tau

        # Critic network
        self.inputs, self.action, self.outputs = self.create_critic_network()
        self.net_params = tf.trainable_variables()[num_actor_vars:]

        # Target network
        self.target_inputs, self.target_action, self.target_outputs = self.create_critic_network()
        self.target_net_params = tf.trainable_variables()[len(self.net_params) + num_actor_vars:]

        # Op for periodically updating target network with online network weights
        self.update_target_net_params = \
            [self.target_net_params[i].assign(tf.multiply(self.net_params[i], self.tau) +
                                              tf.multiply(self.target_net_params[i], 1. - self.tau))
             for i in range(len(self.target_net_params))]

        # Network target (y_i)
        # Obtained from the target networks
        self.predicted_q_value = tf.placeholder(tf.float32, [None, 1])

        # Define loss and optimization Op
        self.loss = tflearn.mean_square(self.predicted_q_value, self.outputs)
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

        # Get the gradient of the critic w.r.t. the action
        self.action_grads = tf.gradients(self.outputs, self.action)

    """
    Build the neural network according to specifications from constructor.
    This function is to be used internally, and should not be called from
    outside the class.

    return: input and output tensorflow objects, plus outputs scaled to the
        action_bound.
    """
    def create_critic_network(self):
        inputs = tflearn.input_data(shape=[None, self.state_dim])
        action = tflearn.input_data(shape=[None, self.action_dim])
        net = tflearn.fully_connected(inputs, 400, activation='relu')
        # Add the action tensor in the 2nd hidden layer
        # Use two temp layers to get the corresponding weights and biases
        t1 = tflearn.fully_connected(net, 300)
        t2 = tflearn.fully_connected(action, 300)
        net = tflearn.activation(tf.matmul(net, t1.W) + tf.matmul(action, t2.W) + t2.b, activation='relu')
        # Linear layer connected to 1 output representing Q(s,a)
        # Weights are init to Uniform[-3e-3, 3e-3]
        weight_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        outputs = tflearn.fully_connected(net, 1, weights_init=weight_init)
        return inputs, action, outputs

    """
    Train method to update the network, by back-propagating new Q-values such
    that the inputs and action produce these Q-values.

    inputs: state observation information.
    action: actions taken in these states.
    predicted_q_value: desired Q-values for the state-action pairs.
    return: sess object information.
    """
    def train(self, inputs, action, predicted_q_value):
        return self.sess.run([self.outputs, self.optimize], feed_dict={
            self.inputs: inputs,
            self.action: action,
            self.predicted_q_value: predicted_q_value
        })

    """
    Provides Q-value for state-action pairs.

    inputs: state observation information.
    action: actions taken in these states.
    return: Q-values for the state-action pairs.
    """
    def predict(self, inputs, action):
        return self.sess.run(self.outputs, feed_dict={
            self.inputs: inputs,
            self.action: action
        })

    """
    As above, using the target network.

    inputs: state observation information.
    action: actions taken in these states.
    return: Q-values for the state-action pairs.
    """
    def predict_target(self, inputs, action):
        return self.sess.run(self.target_outputs, feed_dict={
            self.target_inputs: inputs,
            self.target_action: action
        })

    """
    Output the gradient of the critic network with respect to the action and
    state information.

    inputs: state observation information.
    action: actions taken in these states.
    return: the action gradients.
    """
    def action_gradients(self, inputs, action):
        return self.sess.run(self.action_grads, feed_dict={
            self.inputs: inputs,
            self.action: action
        })

    """
    Update target network.
    """
    def update_target_network(self):
        self.sess.run(self.update_target_net_params)
