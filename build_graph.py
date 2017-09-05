import tensorflow as tf
import lightsaber.tensorflow.util as util


def build_train(encode, num_actions, optimizer, dnds, batch_size=32,
                grad_norm_clipping=10.0, gamma=1.0, scope='deepq', reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        obs_t_input = tf.placeholder(tf.float32, [None, 84, 84, 4], name='obs_t')
        act_t_ph = tf.placeholder(tf.int32, [None], name='action')
        target_values = tf.placeholder(tf.float32, [None], name='value')

        encoded_state = encode(obs_t_input, num_actions, scope='encode')
        encode_vars = util.scope_vars(util.absolute_scope_name('encode'))
        q_values = []
        for i, dnd in enumerate(dnds):
            keys, values = tf.py_func(dnd.lookup, [encoded_state], [tf.float32, tf.float32])
            square_diff = tf.square(keys - tf.expand_dims(encoded_state, 1))
            distances = tf.reduce_sum(square_diff, axis=2) + 1e-3
            weights = 1 / distances
            normalized_weights = weights / tf.reduce_sum(weights, axis=1)
            q_values.append(tf.reduce_sum(normalized_weights * values, axis=1))
        q_t = tf.transpose(q_values)

        q_t_selected = tf.reduce_sum(q_t * tf.one_hot(act_t_ph, num_actions), axis=1)

        errors = tf.reduce_sum(tf.square(target_values - q_t_selected))

        gradients = optimizer.compute_gradients(errors, var_list=encode_vars)
        for i, (grad, var) in enumerate(gradients):
            if grad is not None:
                gradients[i] = (tf.clip_by_norm(grad, grad_norm_clipping), var)
        optimize_expr = optimizer.apply_gradients(gradients)

        actions = tf.reshape(tf.argmax(q_t, axis=1), [-1])
        act = util.function(inputs=[obs_t_input], outputs=[actions, encoded_state])

        train = util.function(
            inputs=[
                obs_t_input, act_t_ph, target_values
            ],
            outputs=errors,
            updates=[optimize_expr]
        )

        q_values = util.function([obs_t_input], q_t)

        return act, train, q_values
