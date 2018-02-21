import tensorflow as tf


class DND:
    '''
    TensorFlow impelementation of DND.
    DND will be created per actions.
    '''
    def __init__(self, keysize=512, capacity=10 ** 5, p=50, lr=0.1):
        self.capacity = capacity
        self.lr = lr
        self.p = p
        self.keysize = keysize  # TODO: check input size

    def _init_vars(self):
        with tf.name_scope('MEMORY_MODULE'):
            # TODO: change
            with tf.device('/cpu:0'):
                self.curr_epsize = tf.Variable(self.p, dtype=tf.float32)
                self.memory_keys = tf.Variable(
                    tf.zeros([self.capacity, self.keysize], dtype=tf.float32),
                    name='KEYS'
                )
                self.memory_values = tf.Variable(
                    tf.zeros([self.capacity], dtype=tf.float32),
                    name='VALUES'
                )
                self.memory_ages = tf.Variable(
                    tf.zeros([self.capacity], dtype=tf.float32),
                    name='AGES'
                )

    def _build_network(self, readerin, hin, vin, epsize):
        self.writer = self._build_writer(hin, vin, epsize)
        self.reader = self._build_reader(readerin, epsize)
        return self.reader, self.writer

    def _build_reader(self, h, epsize):
        with tf.name_scope('lookup'):
            keys = self.memory_keys[:epsize]
            tiled_keys = tf.tile([keys], [tf.shape(h)[0], 1, 1])
            expanded_h = tf.expand_dims(h, axis=1)
            distances = tf.reduce_sum(
                tf.square(
                    tiled_keys - tf.tile(
                        expanded_h, [1, epsize, 1]
                    )
                ), axis=2
            )
            values = self.memory_values[:epsize]

            # negate distances to get the k closest keys
            _, indices = tf.nn.top_k(-distances, k=self.p)  # indecies (?, 10)
            # distances (?, ?) batchsize, memsize
            # get p distances
            hit_keys = tf.nn.embedding_lookup(keys, indices)
            hit_values = tf.nn.embedding_lookup(values, indices)
            flatten_indices = tf.reshape(indices, [-1])  # batch * self.p
            unique_indicies, _ = tf.unique(flatten_indices)
            update_ages = tf.group(*[
                # increment ages
                tf.assign(self.memory_ages, self.memory_ages + 1),
                # reset hit ages
                tf.scatter_update(
                    self.memory_ages, unique_indicies,
                    tf.zeros([self.p], dtype=tf.float32)
                )
                # tf.assign(hit_ages, 0)
            ])
        return hit_keys, hit_values, update_ages

    def _build_writer(self, hin, vin, epsize):
        with tf.name_scope('write'):
            # memory_keys (capacity, keysize)
            # hin (keysize) -> (capacity, keysize)  broadcasted
            with tf.name_scope('GETDIST'):
                distvec = self.memory_keys - hin
                distance = tf.norm(distvec, axis=1)  # (capacity)
                index = tf.argmin(distance, axis=0)  # scalar
                mindist = distance[index]
    
            with tf.name_scope('BODY1'):
                # check if distance is the same
                new_value = self.lr * (vin - self.memory_values[index])
                update_val = tf.assign(self.memory_values[index], new_value)
    
            with tf.name_scope('BODY2'):
                with tf.name_scope('IF'):
                    update_age = tf.assign(  # TODO: remove this
                        self.memory_ages[epsize - 1], 0
                    )
                    key_append = tf.assign(
                        self.memory_keys[epsize - 1], hin
                    )
                    val_append = tf.assign(
                        self.memory_values[epsize - 1], vin
                    )
                    # inc update should be in order
                    deps = tf.group(
                        *[update_age, key_append, val_append]
                    )
    
                    with tf.control_dependencies([deps]):
                        grouped_appends = tf.group(
                            tf.assign_add(
                                self.curr_epsize, 1
                            )
                        )
    
                with tf.name_scope('ELSE'):
                    oldest_idx = tf.argmax(self.memory_ages)
                    key_update = tf.assign(
                        self.memory_keys[oldest_idx], hin
                    )
                    val_update = tf.assign(
                        self.memory_values[oldest_idx], vin
                    )
                    grouped_updates = tf.group(
                        *[key_update, val_update]
                    )
    
            cond_dist_eq = tf.cond(
                tf.equal(mindist, 0),
                lambda: update_val, lambda: .0,
                name='COND1'
            )
    
            with tf.control_dependencies([cond_dist_eq]):
                # pev operation followed by below
                cond_capa_less = tf.cond(
                    tf.less(epsize, self.capacity),
                    lambda: grouped_updates, lambda: grouped_appends,
                    name='COND2'
                )
            end_node = cond_capa_less

        return end_node
