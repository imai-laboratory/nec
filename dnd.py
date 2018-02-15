# import numpy as np
import tensorflow as tf
# from sklearn.neighbors.kd_tree import KDTree
# from collections import deque


class DND:
    '''
    TensorFlow impelementation of DND.
    DND will be created per actions.
    '''
    def __init__(self, keysize=512, capacity=10 ** 5, p=10, lr=0.1):
        self.capacity = capacity
        self.lr = lr
        self.p = p
        self.keysize = keysize  # TODO: check input size

    def _init_vars(self):
        with tf.name_scope('MEMORY_MODULE'):
            self.curr_epsize = tf.Variable(
                50, dtype=tf.int32
            )
            self.memory_keys = tf.Variable(
                tf.zeros([self.capacity, self.keysize], dtype=tf.float32),
                name='KEYS'
            )
            self.memory_values = tf.Variable(
                tf.zeros([self.capacity], dtype=tf.float32),
                name='VALUES'
            )
            self.memory_ages = tf.Variable(
                tf.zeros([self.capacity], dtype=tf.int32),
                name='AGES'
            )

    def _build_network(self, readerin, hin, vin, epsize):
        self.writer = self._build_writer(hin, vin, epsize)
        self.reader = self._build_reader(readerin, epsize)
        return self.reader, self.writer

    def ex_gather_2d(self, data, indices_mat):
        # data (batch, memsize, 160)
        shape = tf.shape(data)
        batch_size = shape[0]
        mem_size = tf.cast(shape[1], dtype=tf.float32)
        flattened_data = tf.reshape(data, [-1])  # dim1
        flatten_indices = tf.reshape(indices_mat, [-1])
        expanded_offsets = tf.expand_dims(
            tf.range(tf.cast(batch_size, dtype=tf.float32), dtype=tf.float32)\
            * mem_size, 1
        )
        tiled_offsets = tf.tile(expanded_offsets, [1, self.p])
        flattened_offsets = tf.reshape(tiled_offsets, [-1])  # dim1
        gathered_data = tf.gather(
            flattened_data, flatten_indices\
            + tf.cast(flattened_offsets, tf.int32)
        )
        reshaped_data = tf.reshape(
            gathered_data, [batch_size, self.p]
        )
        return reshaped_data

    def ex_gather_3d(self, data, indices_mat):
        # data (batch, memsize, 160)
        shape = tf.shape(data)
        batch_size = shape[0]
        mem_size = tf.cast(shape[1], dtype=tf.float32)
        flattened_data = tf.reshape(data, [-1, self.keysize])  # dim1
        flatten_indices = tf.reshape(indices_mat, [-1])
        expanded_offsets = tf.expand_dims(
            tf.range(tf.cast(batch_size, dtype=tf.float32),
                     dtype=tf.float32) * mem_size, 1
        )
        tiled_offsets = tf.tile(expanded_offsets, [1, self.p])
        flattened_offsets = tf.reshape(tiled_offsets, [-1])  # dim1
        gathered_data = tf.gather(
            flattened_data, flatten_indices\
            + tf.cast(flattened_offsets, tf.int32)
        )
        reshaped_data = tf.reshape(
            gathered_data, [batch_size, self.p, self.keysize]
        )
        return reshaped_data

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
            tiled_values = tf.tile([values], [tf.shape(h)[0], 1])

            # negate distances to get the k closest keys
            _, indices = tf.nn.top_k(-distances, k=self.p)  # indecies (?, 10)
            # distances (?, ?) batchsize, memsize
            # get p distances
            hit_keys = self.ex_gather_3d(tiled_keys, indices)
            hit_values = self.ex_gather_2d(tiled_values, indices)
            flatten_indices = tf.reshape(indices, [-1])  # batch * self.p
            unique_indicies, _ = tf.unique(flatten_indices)
            update_ages = tf.group(*[
                # increment ages
                tf.assign(self.memory_ages, self.memory_ages + 1),
                # reset hit ages
                tf.scatter_update(
                    self.memory_ages, unique_indicies,
                    tf.zeros([self.p], dtype=tf.int32)
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
