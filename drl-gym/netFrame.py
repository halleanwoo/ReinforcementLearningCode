import tensorflow as tf
import tensorflow.contrib.layers as layers
from argument import args

# net_frame using for creating Q & target network
def net_frame_mlp(hiddens, inpt, num_actions, scope, reuse=False, layer_norm=False):
    with tf.variable_scope(scope, reuse=reuse):
        out = inpt
        for hidden in hiddens:
            out = layers.fully_connected(out, num_outputs=hidden, activation_fn=None)
            if layer_norm:
                out = layers.layer_norm(out, center=True, scale=True)
            out = tf.nn.relu(out)
        q_out = layers.fully_connected(out, num_outputs=num_actions, activation_fn=None)
        return q_out

def net_frame_cnn_to_mlp(convs, hiddens, inpt, num_actions, scope, dueling=False, reuse=False, layer_norm=False):
    with tf.variable_scope(scope, reuse=reuse):
        out = inpt
        with tf.variable_scope("convnet"):
            for num_outputs, kernel_size, stride in convs:
                out = layers.convolution2d(out,
                                           num_outputs=num_outputs,
                                           kernel_size=kernel_size,
                                           stride=stride,
                                           activation_fn=tf.nn.relu)
        conv_out = layers.flatten(out)
        with tf.variable_scope("action_value"):
            action_out = conv_out
            for hidden in hiddens:
                action_out = layers.fully_connected(action_out, num_outputs=hidden, activation_fn=None)
                if layer_norm:
                    action_out = layers.layer_norm(action_out, center=True, scale=True)    # normlization:这里使用的不是BN，而是LN
                action_out = tf.nn.relu(action_out)
            action_scores = layers.fully_connected(action_out, num_outputs=num_actions, activation_fn=None)

        if dueling:
            with tf.variable_scope("state_value"):
                state_out = conv_out
                for hidden in hiddens:
                    state_out = layers.fully_connected(state_out, num_outputs=hidden, activation_fn=None)
                    if layer_norm:
                        state_out = layers.layer_norm(state_out, center=True, scale=True)
                    state_out = tf.nn.relu(state_out)
                state_score = layers.fully_connected(state_out, num_outputs=1, activation_fn=None)
            action_scores_mean = tf.reduce_mean(action_scores, 1)
            action_scores_centered = action_scores - tf.expand_dims(action_scores_mean, 1)
            q_out = state_score + action_scores_centered
        else:
            q_out = action_scores
        return q_out


# ref: break-out-master        
def build_net(s, var_scope, dueling=0):
    with tf.variable_scope(var_scope):
        with tf.variable_scope('conv1'):
            W1 = init_W(shape=[8, 8, 4, 32])
            b1 =  init_b(shape=[32])
            conv1 =  conv2d(s, W1, strides=4)
            h_conv1 = tf.nn.relu(tf.nn.bias_add(conv1, b1))

        # with tf.name_scope('max_pool1'):
        #   h_pool1 =  max_pool(h_conv1)

        with tf.variable_scope('conv2'):
            W2 =  init_W(shape=[4, 4, 32, 64])
            b2 =  init_b(shape=[64])
            conv2 =  conv2d(h_conv1, W2, strides=2)
            h_conv2 = tf.nn.relu(tf.nn.bias_add(conv2, b2))

        with tf.variable_scope('conv3'):
            W3 =  init_W(shape=[3, 3, 64, 64])
            b3 =  init_b(shape=[64])
            conv3 =  conv2d(h_conv2, W3, strides=1)
            h_conv3 = tf.nn.relu(tf.nn.bias_add(conv3, b3))

        h_flatten = tf.reshape(h_conv3, [-1, 3136])
        
        with tf.variable_scope('fc1'):
            W_fc1 =  init_W(shape=[3136, 512])
            b_fc1 =  init_b(shape=[512])
            fc1 = tf.nn.bias_add(tf.matmul(h_flatten, W_fc1), b_fc1)

        if not  dueling:
            with tf.variable_scope('fc2'):
                h_fc1 = tf.nn.relu(fc1)
                W_fc2 =  init_W(shape=[512, 4])
                b_fc2 =  init_b(shape=[4])
                out = tf.nn.bias_add(tf.matmul(h_fc1, W_fc2), b_fc2, name='Q')
        else:
            with tf.variable_scope('Value'):
                h_fc1_v = tf.nn.relu(fc1)
                W_v =  init_W(shape=[512, 1])
                b_v =  init_b(shape=[1])
                V = tf.nn.bias_add(tf.matmul(h_fc1_v, W_v), b_v, name='V')

            with tf.variable_scope('Advantage'):
                h_fc1_a = tf.nn.relu(fc1)
                W_a =  init_W(shape=[512, 4])
                b_a =  init_b(shape=[4])
                A = tf.nn.bias_add(tf.matmul(h_fc1_a, W_a), b_a, name='A')
          
            with tf.variable_scope('Q'):
                out =  V + ( A - tf.reduce_mean( A, axis=1, keep_dims=True))
    return out

def init_W(shape, name='weights', w_initializer=tf.truncated_normal_initializer(0, 1e-2)):

    return tf.get_variable(
      name=name,
      shape=shape, 
      initializer=w_initializer)

def init_b(shape, name='biases',  b_initializer = tf.constant_initializer(1e-2)):

    return tf.get_variable(
      name=name,
      shape=shape,
      initializer=b_initializer)

def conv2d(x, kernel, strides=4):

    return tf.nn.conv2d(
      input=x, 
      filter=kernel, 
      strides=[1, strides, strides, 1], 
      padding="VALID")

def max_pool(x, ksize=2, strides=2):
    return tf.nn.max_pool(x, 
      ksize=[1, ksize, ksize, 1], 
      strides=[1, strides, strides, 1], 
      padding="SAME")
