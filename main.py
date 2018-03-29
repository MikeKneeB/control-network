import train, simple_m_env
import tensorflow as tf
import tflearn
import actor
import critic

if __name__ == '__main__':
    with tf.Session() as sess:
        # Make our environment.
        env = simple_m_env.MissileEnv()

        # Get environment params, for building networks etc.
        state_dim = 3
        action_dim = 3
        max_action = 1.

        print('{} {} {}'.format(state_dim, action_dim, max_action))

        # Build our actor and critic agents.
        actor_model = actor.ActorNetwork(sess, state_dim, action_dim, max_action, 0.0001, 0.001)
        critic_model = critic.CriticNetwork(sess, state_dim, action_dim, max_action, 0.001, 0.001, actor_model.get_num_trainable_vars())

        # Train.
        train.train(sess, actor_model, critic_model, env, state_dim, action_dim,
        max_action, epochs=300, run_length=400, render=True, envname='pendulum', decay=0.995)

        input('Training complete, press enter to continue to test.')

        # Test.
        #test(sess, actor_model, critic_model, env, state_dim, action_dim, epochs=10, run_length=300, filename='out.dat')
