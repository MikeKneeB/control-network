import train
import tensorflow as tf
import gym
import tflearn
import actor
import critic

if __name__ == '__main__':
    with tf.Session() as sess:
        # Make our environment.
        env = gym.make('Pendulum-v0')
        env.seed(int(time.time()))

        # Get environment params, for building networks etc.
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        max_action = env.action_space.high

        # Build our actor and critic agents.
        actor_model = actor.ActorNetwork(sess, state_dim, action_dim, max_action, 0.0001, 0.001)
        critic_model = critic.CriticNetwork(sess, state_dim, action_dim, max_action, 0.001, 0.001, actor_model.get_num_trainable_vars())

        # Train.
        train.train(sess, actor_model, critic_model, env, state_dim, action_dim,
        max_action, epochs=300, run_length=200, render=True, envname='pendulum', decay=0.98)

        input('Training complete, press enter to continue to test.')

        # Test.
        train.test(sess, actor_model, critic_model, env, state_dim, action_dim, epochs=10, run_length=300, filename='out.dat')
