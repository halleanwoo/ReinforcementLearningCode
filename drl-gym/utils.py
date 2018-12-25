import tensorflow as tf
import random
import numpy as np

def set_random_seed(SEED=0):
    tf.set_random_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

def huber_loss(x, delta=1.0):
    """Reference: https://en.wikipedia.org/wiki/Huber_loss"""
    return tf.where(
        tf.abs(x) < delta,
        tf.square(x) * 0.5,
        delta * (tf.abs(x) - 0.5 * delta)
    )


def build_rmsprop_optimizer(learning_rate, rmsprop_decay, rmsprop_constant, gradient_clip, version, loss):

    with tf.name_scope('rmsprop'):
        optimizer = None
        if version == 'rmsprop':
            optimizer = tf.train.RMSPropOptimizer(learning_rate, decay=rmsprop_decay, momentum=0.0, epsilon=rmsprop_constant)
        elif version == 'graves_rmsprop':
            optimizer = tf.train.GradientDescentOptimizer(learning_rate)

        grads_and_vars = optimizer.compute_gradients(loss)
        grads = [gv[0] for gv in grads_and_vars]
        params = [gv[1] for gv in grads_and_vars]
        # print(grads)
        if gradient_clip > 0:
            grads = tf.clip_by_global_norm(grads, gradient_clip)[0]

        grads = [grad for grad in grads if grad != None]

        if version == 'rmsprop':
            return optimizer.apply_gradients(zip(grads, params))
        elif version == 'graves_rmsprop':
            square_grads = [tf.square(grad) for grad in grads if grad != None]

            avg_grads = [tf.Variable(tf.zeros(var.get_shape())) for var in params]
            avg_square_grads = [tf.Variable(tf.zeros(var.get_shape())) for var in params]

            update_avg_grads = [grad_pair[0].assign((rmsprop_decay * grad_pair[0]) + ((1 - rmsprop_decay) * grad_pair[1])) 
            for grad_pair in zip(avg_grads, grads)]

            update_avg_square_grads = [grad_pair[0].assign((rmsprop_decay * grad_pair[0]) + ((1 - rmsprop_decay) * tf.square(grad_pair[1]))) 
            for grad_pair in zip(avg_square_grads, grads)]
            avg_grad_updates = update_avg_grads + update_avg_square_grads

            rms = [tf.sqrt(avg_grad_pair[1] - tf.square(avg_grad_pair[0]) + rmsprop_constant)
            for avg_grad_pair in zip(avg_grads, avg_square_grads)]

            rms_updates = [grad_rms_pair[0] / grad_rms_pair[1] for grad_rms_pair in zip(grads, rms)]
            train = optimizer.apply_gradients(zip(rms_updates, params))

            return tf.group(train, tf.group(*avg_grad_updates))