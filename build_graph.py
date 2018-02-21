import tensorflow as tf
import lightsaber.tensorflow.util as util
from tensorflow.python.client import timeline


def build_train(encode, num_actions, optimizer, dnds,
                options, scope='DEEPQ',
                run_options=None, run_metadata=None,
                reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        # TODO: remove
        assert run_options is not None
        assert run_metadata is not None

        obs_t_input = tf.placeholder(tf.float32, [None, 84, 84, 4], name='obs_t')
        act_t_ph = tf.placeholder(tf.int32, [None], name='action')
        target_values = tf.placeholder(tf.float32, [None], name='value')

        encoded_state = encode(obs_t_input, num_actions, scope='encode')
        encode_vars = util.scope_vars(util.absolute_scope_name('encode'))
        q_values = []
        writs = []

        # Place holders for DND
        hin = tf.placeholder(
            tf.float32, [options.hin_size], name='key'
        )
        vin = tf.placeholder(
            tf.float32, [], name='value'
        )

        epsize = tf.placeholder(
            tf.int32, [num_actions], name='epsize'
        )

        for i, dnd in enumerate(dnds):
            with tf.name_scope('DND'):
                reader, writer = dnd._build_network(
                    encoded_state, hin, vin, epsize[i]
                )
                keys, values, update_ages = reader
                expanded_encode = tf.tile(
                    tf.expand_dims(encoded_state, axis=1),
                    [1, dnd.p, 1]
                )
                distances = tf.reduce_sum(
                    # tf.square(tf.stop_gradient(keys) - expanded_encode),
                    tf.square(keys - expanded_encode),
                    axis=2
                )
                k = 1.0 / (distances + 10e-20)
                weights = (k /
                           tf.reshape(
                               tf.reduce_sum(k, axis=1),
                               [-1, 1])
                           )
                q_values.append(tf.reduce_sum(weights * values, axis=1))
                writs.append(writer)

        # get actions
        q_t = tf.transpose(q_values)

        q_t_selected = tf.reduce_sum(
            q_t * tf.one_hot(act_t_ph, num_actions), axis=1
        )

        # GRADIENTS
        errors = tf.reduce_sum(tf.square(target_values - q_t_selected))
        gradients = optimizer.compute_gradients(errors, var_list=encode_vars)
        for i, (grad, var) in enumerate(gradients):
            if grad is not None:
                gradients[i] = (tf.clip_by_norm(grad, options.grad_norm_clipping), var)
        optimize_expr = optimizer.apply_gradients(gradients)

        actions = tf.reshape(tf.argmax(q_t, axis=1), [-1])
        act = util.function(
            inputs=[obs_t_input, epsize],
            outputs=[actions, q_t, encoded_state],
            options=run_options,
            run_metadata=run_metadata
        )

        train = util.function(
            inputs=[
                obs_t_input, act_t_ph, target_values, epsize
            ],
            outputs=errors,
            updates=[optimize_expr],
            options=run_options,
            run_metadata=run_metadata
        )

        writers = [
            util.function(
                inputs=[hin, vin, epsize],
                outputs=w,
            )
            for w in writs
        ]

        return act, writers, train
