import tensorflow as tf


def build_train(encode,
                num_actions,
                optimizer,
                dnds,
                state_shape,
                key_size,
                grad_clipping=10.0,
                scope='nec',
                run_options=None,
                run_metadata=None,
                reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        # placeholders for CNN
        obs_t_input = tf.placeholder(tf.float32, [None] + state_shape, name='obs_t')
        act_t_ph = tf.placeholder(tf.int32, [None], name='action')
        target_values = tf.placeholder(tf.float32, [None], name='value')

        encoded_state = encode(obs_t_input, scope='encode')
        encode_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, '{}/encode'.format(scope))
        q_values = []
        writs = []

        # placeholders for DND
        hin = tf.placeholder(tf.float32, [key_size], name='key')
        vin = tf.placeholder(tf.float32, [], name='value')
        epsize = tf.placeholder(tf.int32, [num_actions], name='epsize')

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
                distances = tf.reduce_sum(tf.square(keys - expanded_encode), axis=2)
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

        # td error
        error = tf.reduce_sum(tf.square(target_values - q_t_selected))

        # gradients
        trained_vars = encode_vars
        for i in range(num_actions):
            trained_vars += tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, 'dnd{}/KEYS'.format(i))
        gradients = optimizer.compute_gradients(error, var_list=trained_vars)
        for i, (grad, var) in enumerate(gradients):
            if grad is not None:
                gradients[i] = (tf.clip_by_norm(grad, grad_clipping), var)
        optimize_expr = optimizer.apply_gradients(gradients)

        actions = tf.reshape(tf.argmax(q_t, axis=1), [-1])
        def act(obs, ep):
            feed_dict = {
                obs_t_input: obs,
                epsize: ep
            }
            return tf.get_default_session().run(
                [actions, q_t, encoded_state], feed_dict=feed_dict,
                options=run_options, run_metadata=run_metadata)

        def train(obs, act, target, ep):
            feed_dict = {
                obs_t_input: obs,
                act_t_ph: act,
                target_values: target,
                epsize: ep
            }
            error_val, _ = tf.get_default_session().run(
                [error, optimize_expr], feed_dict=feed_dict,
                options=run_options, run_metadata=run_metadata)
            return error_val

        writers = []
        for i in range(num_actions):
            def writer_func(index, h, v, ep):
                feed_dict = {
                    hin: h,
                    vin: v,
                    epsize: ep
                }
                sess = tf.get_default_session()
                return sess.run(writs[index], feed_dict=feed_dict)
            writers.append(lambda h, v, ep, index=i: writer_func(index, h, v, ep))

        return act, writers, train
