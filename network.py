import tensorflow as tf
import tensorflow.contrib.layers as layers


def _make_cnn(convs, hiddens, inpt, num_actions, scope, reuse=None):
    ''' creates convlayers
    ARGS:
        convs (list): a list of tuple(s) which includes layer settings.
        each tuple represents (output_size, kernel_size, stride)
        hiddens (list): a list of integers which includes FC layer settings.
        each integer represents the number of hidden units in a FC layer.
    '''
    with tf.variable_scope(scope, reuse=reuse):
        out = inpt
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
        conv_out = layers.flatten(out)
        with tf.variable_scope('fully_connected'):
            encode = conv_out
            for hidden in hiddens:
                encode = layers.fully_connected(encode, num_outputs=hidden, activation_fn=None)
        return encode


def make_cnn(convs, hiddens):
    ''' returns a function to create convlayers
    ARGS:
        convs (list): a list of tuple(s) which includes layer settings.
        each tuple represents (output_size, kernel_size, stride)
        hiddens (list): a list of integers which includes FC layer settings.
        each integer represents the number of hidden units in a FC layer.
    '''
    return lambda *args, **kwargs: _make_cnn(convs, hiddens, *args, **kwargs)
