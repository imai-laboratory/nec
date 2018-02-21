import tensorflow as tf
import tensorflow.contrib.layers as layers


def _make_network(convs, hiddens, key_size, inpt, scope):
    ''' creates convlayers
    ARGS:
        convs (list): a list of tuple(s) which includes layer settings.
        each tuple represents (output_size, kernel_size, stride).
        if list is None then conv layers will be skipped.
        hiddens (list): a list of integers which includes FC layer settings.
        each integer represents the number of hidden units in a FC layer.
    '''
    with tf.variable_scope(scope):
        out = inpt  # for skipping conv layers
        with tf.variable_scope('convnet'):
            for num_outputs, kernel_size, stride in convs:
                out = layers.convolution2d(
                    out,
                    num_outputs=num_outputs,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding='SAME',  # using zero padding
                    activation_fn=tf.nn.relu
                )
        out = layers.flatten(out)
        with tf.variable_scope('fully_connected'):
            for hidden in hiddens:
                out = layers.fully_connected(
                    out,
                    num_outputs=hidden,
                    activation_fn=tf.nn.relu
                )
        # encoded key layer
        encode = layers.fully_connected(
            out,
            num_outputs=key_size,
            activation_fn=None,
            scope='encode'
        )
        return encode


def make_network(convs, hiddens, key_size):
    ''' returns a function to create convlayers
    ARGS:
        convs (list): a list of tuple(s) which includes layer settings.
        each tuple represents (output_size, kernel_size, stride)
        hiddens (list): a list of integers which includes FC layer settings.
        each integer represents the number of hidden units in a FC layer.
    '''
    return lambda *args, **kwargs: _make_network(convs, hiddens, key_size, *args, **kwargs)
