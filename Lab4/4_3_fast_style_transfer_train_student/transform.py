import tensorflow as tf, pdb

WEIGHTS_INIT_STDEV = .1

def net(image, type=0):
    conv1 = _conv_layer(image, 32, 9, 1, type)
    conv2 = _conv_layer(conv1, 64 ,3, 2, type)
    conv3 = _conv_layer(conv2, 128 ,3, 2, type)
    residual1 = _residual_block(conv3, 3, type=1)
    residual2 = _residual_block(residual1, 3, type=1)
    residual3 = _residual_block(residual2, 3, type=1)
    residual4 = _residual_block(residual3, 3, type=1)
    residual5 = _residual_block(residual4, 3, type=1)
    conv_transpose1 = _conv_tranpose_layer(residual5, 64, 3, 2)
    conv_transpose2 = _conv_tranpose_layer(conv_transpose1, 32, 3, 2)
    conv4 = _conv_layer(conv_transpose2, 3, 9, 1)

    preds = (tf.nn.tanh(conv4) + 1) * 255 / 2.0

    return preds

def _conv_layer(net, num_filters, filter_size, strides, relu=True, type=0):
    weights_init = _conv_init_vars(net, num_filters, filter_size)
    bias_init = tf.Variable(tf.zeros([num_filters], dtype=tf.float32))
    strides_shape = [1] + list([strides]) * 2 + [1]
    net = tf.nn.bias_add(tf.nn.conv2d(net, weights_init, strides=strides_shape, padding='SAME'), bias_init)

    if type == 0:
        net = _batch_norm(net)
    elif type == 1:
        net = _instance_norm(net)

    if relu:
        net = tf.nn.relu(net)

    return net

def _conv_tranpose_layer(net, num_filters, filter_size, strides, type=0):
    weights_init = _conv_init_vars(net, num_filters, filter_size, transpose=True)
    bias_init = tf.Variable(tf.zeros([num_filters], dtype=tf.float32))
    batch, rows, cols, channels = [i.value for i in net.get_shape()]
    strides_shape = [1] + list([strides]) * 2 + [1]
    new_shape = [batch, rows * strides, cols * strides, num_filters]
    output_shape = tf.stack(new_shape)
    net = tf.nn.bias_add(tf.nn.conv2d_transpose(net, weights_init, output_shape=output_shape, strides=strides_shape, padding='SAME'), bias_init)
    
    if type == 0:
        net = _batch_norm(net)
    elif type == 1:
        net = _instance_norm(net)
    
    net = tf.nn.relu(net)

    return net

def _residual_block(net, filter_size=3, type=0):
    x_shape = net.get_shape()
    fx = _conv_layer(_conv_layer(net, 128, filter_size, 1, relu=True), 128, filter_size, 1, relu=False)
    fx_shape = fx.get_shape()

    if (x_shape[0] == fx_shape[0] and x_shape[1] == fx_shape[1]):
        net = tf.nn.relu(fx + net)
    else:
        net = tf.nn.relu(fx + _conv_layer(net, 128, 1, 1, relu=False))

    return net

def _batch_norm(net, train=True):
    batch, rows, cols, channels = [i.value for i in net.get_shape()]
    axes=list(range(len(net.get_shape())-1))
    mu, sigma_sq = tf.nn.moments(net, axes, keep_dims=True)
    var_shape = [channels]
    shift = tf.Variable(tf.zeros(var_shape))
    scale = tf.Variable(tf.ones(var_shape))
    epsilon = 1e-3
    return tf.nn.batch_normalization(net, mu, sigma_sq, shift, scale, epsilon)

def _instance_norm(net, train=True):
    batch, rows, cols, channels = [i.value for i in net.get_shape()]
    var_shape = [channels]
    mu, sigma_sq = tf.nn.moments(net, [1,2], keep_dims=True)
    shift = tf.Variable(tf.zeros(var_shape))
    scale = tf.Variable(tf.ones(var_shape))
    epsilon = 1e-3
    normalized = (net-mu)/(sigma_sq + epsilon)**(.5)
    return scale * normalized + shift

def _conv_init_vars(net, out_channels, filter_size, transpose=False):
    _, rows, cols, in_channels = [i.value for i in net.get_shape()]
    if not transpose:
        weights_shape = [filter_size, filter_size, in_channels, out_channels]
    else:
        weights_shape = [filter_size, filter_size, out_channels, in_channels]

    weights_init = tf.Variable(tf.truncated_normal(weights_shape, stddev=WEIGHTS_INIT_STDEV, seed=1), dtype=tf.float32)
    return weights_init
