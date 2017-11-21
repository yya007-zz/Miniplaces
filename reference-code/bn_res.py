
import bn_read as rb
import tensorflow as tf
from bn_para import *

learning_rate = 0.001
training_iters = 100000
batch_size = 50
display_step = 20
save_step = 2000
val_step = 20

n_input = 126*126*3
n_classes = 100

x = tf.placeholder(tf.float32, [batch_size,126,126,3])
y = tf.placeholder(tf.int32, [batch_size])
train_phase = tf.placeholder(tf.bool, name='train_phase')
#To tell batch_norm whether to update global mean and var

def top_k_error(predictions, labels, k):
    in_top1 = tf.to_float(tf.nn.in_top_k(predictions, labels, k))
    num_correct = tf.reduce_sum(in_top1)
    return (float(batch_size) - num_correct) / float(batch_size)

def conv2d(name, l_input, w):
    #tf.nn.conv2d([patch,hight,width,channel], [filter_height, filter_width, in_channels, out_channels], strides=[patch,hight,width,channel])
    return tf.nn.conv2d(l_input, w, strides=[1, 1, 1, 1], padding='SAME', name=name)
def conv2dp(name, l_input, w):
    #tf.nn.conv2d([patch,hight,width,channel], [filter_height, filter_width, in_channels, out_channels], strides=[patch,hight,width,channel])
    return tf.nn.conv2d(l_input, w, strides=[1, 2, 2, 1], padding='SAME', name=name)
def bias(name, l_input,b):
    return tf.nn.bias_add(l_input,b,name=name)
def relu(name, l_input):
    return tf.nn.relu(l_input,name=name)
def max_pool(name, l_input, k):
    # ksize[patch,height,width,channel], strides[patch,height,width,channel]
    return tf.nn.max_pool(l_input, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME', name=name)
def avg_pool(name, l_input, k):
    # ksize[patch,height,width,channel], strides[patch,height,width,channel]
    return tf.nn.avg_pool(l_input, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME', name=name)
def cbbr(name, l_input, w, b, bn):
    return relu(name,bias(name,batch_norm(conv2d(name, l_input,w), bn, train_phase, scope='bn'),b))
def cpbbr(name, l_input, w, b, bn):
    return relu(name,bias(name,batch_norm(conv2dp(name, l_input,w), bn, train_phase, scope='bn'),b))
def batch_norm(x, bn, train_phase, scope='bn'):
    """
    Batch normalization on convolutional maps.
    Args:
        x:           Tensor, 4D BHWD input maps
        bn:       integer, depth of input maps
        train_phase: boolean tf.Varialbe, true indicates training phase
        scope:       string, variable scope
    Return:
        normed:      batch-normalized maps
    """
    with tf.variable_scope(scope):
        beta = tf.Variable(tf.constant(0.0, shape=[bn]),name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[bn]),name='gamma', trainable=True)
        batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.5)
        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(train_phase,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed

def res_new(_X, _weights, _biases, _bnorm):

    conv1 = cpbbr('conv1', _X, _weights['wc1'], _biases['bc1'], _bnorm['bn1'])
    #[path,64,64,64]

    conv2a_2a = cbbr('conv2a_2a', conv1, _weights['wc2a_2a'], _biases['bc2a_2a'], _bnorm['bn2a_2a'])
    conv2a_2b = cbbr('conv2a_2b', conv2a_2a, _weights['wc2a_2b'], _biases['bc2a_2b'], _bnorm['bn2a_2b'])
    conv2a_2c = conv2d('conv2a_2c', conv2a_2b, _weights['wc2a_2c'])
    conv2a_2c = bias('conv2a_2c', conv2a_2c, _biases['bc2a_2c'])

    conv2a_1 = conv2d('conv2a_1', conv1, _weights['wc2a_1'])
    conv2a_1 = bias('conv2a_1', conv2a_1, _biases['bc2a_1'])

    conv2a = tf.add(conv2a_1, conv2a_2c)
    conv2a = batch_norm(conv2a, _bnorm['bn2a_1'], train_phase, scope='bn')
    conv2a = relu('conv2a', conv2a)
    #[path,64,64,256]

    conv2b_2a = cbbr('conv2b_2a', conv2a, _weights['wc2b_2a'], _biases['bc2b_2a'], _bnorm['bn2b_2a'])
    conv2b_2b = cbbr('conv2b_2b', conv2b_2a, _weights['wc2b_2b'], _biases['bc2b_2b'], _bnorm['bn2b_2b'])
    conv2b_2c = conv2d('conv2b_2c', conv2b_2b, _weights['wc2b_2c'])
    conv2b_2c = bias('conv2b_2c', conv2b_2c, _biases['bc2b_2c'])

    conv2b = tf.add(conv2a, conv2b_2c)
    conv2b = batch_norm(conv2b, _bnorm['bn2b_2c'], train_phase, scope='bn')
    conv2b = relu('conv2b', conv2b)
    #[path,64,64,256]

    pool2 = avg_pool('pool2', conv2b, k=2)
    #[path,32,32,256]

    conv2c_2a = cpbbr('conv2c_2a', conv2b, _weights['wc2c_2a'], _biases['bc2c_2a'], _bnorm['bn2c_2a'])
    conv2c_2b = cbbr('conv2c_2b', conv2c_2a, _weights['wc2c_2b'], _biases['bc2c_2b'], _bnorm['bn2c_2b'])
    conv2c_2c = conv2d('conv2c_2c', conv2c_2b, _weights['wc2c_2c'])
    conv2c_2c = bias('conv2c_2c', conv2c_2c, _biases['bc2c_2c'])

    conv2c = tf.add(pool2, conv2c_2c)
    conv2c = batch_norm(conv2c, _bnorm['bn2c_2c'], train_phase, scope='bn')
    conv2c = relu('conv2c', conv2c)
    #[path,32,32,256]

    conv3a_2a = cbbr('conv3a_2a', conv2c, _weights['wc3a_2a'], _biases['bc3a_2a'], _bnorm['bn3a_2a'])
    conv3a_2b = cbbr('conv3a_2b', conv3a_2a, _weights['wc3a_2b'], _biases['bc3a_2b'], _bnorm['bn3a_2b'])
    conv3a_2c = conv2d('conv3a_2c', conv3a_2b, _weights['wc3a_2c'])
    conv3a_2c = bias('conv3a_2c', conv3a_2c, _biases['bc3a_2c'])

    conv3a_1 = conv2d('conv3a_1', conv2c, _weights['wc3a_1'])
    conv3a_1 = bias('conv3a_1', conv3a_1, _biases['bc3a_1'])

    conv3a = tf.add(conv3a_1, conv3a_2c)
    conv3a = batch_norm(conv3a, _bnorm['bn3a_1'], train_phase, scope='bn')
    conv3a = relu('conv3a', conv3a)
    #[path,32,32,512]

    conv3b_2a = cbbr('conv3b_2a', conv3a, _weights['wc3b_2a'], _biases['bc3b_2a'], _bnorm['bn3b_2a'])
    conv3b_2b = cbbr('conv3b_2b', conv3b_2a, _weights['wc3b_2b'], _biases['bc3b_2b'], _bnorm['bn3b_2b'])
    conv3b_2c = conv2d('conv3b_2c', conv3b_2b, _weights['wc3b_2c'])
    conv3b_2c = bias('conv3b_2c', conv3b_2c, _biases['bc3b_2c'])

    conv3b = tf.add(conv3a, conv3b_2c)
    conv3b = batch_norm(conv3b, _bnorm['bn3b_2c'], train_phase, scope='bn')
    conv3b = relu('conv3b', conv3b)
    #[path,32,32,512]
    pool3 = avg_pool('pool3', conv3b, k=2)
    #[path,16,16,512]

    conv3c_2a = cpbbr('conv3c_2a', conv3b, _weights['wc3c_2a'], _biases['bc3c_2a'], _bnorm['bn3c_2a'])
    conv3c_2b = cbbr('conv3c_2b', conv3c_2a, _weights['wc3c_2b'], _biases['bc3c_2b'], _bnorm['bn3c_2b'])
    conv3c_2c = conv2d('conv3c_2c', conv3c_2b, _weights['wc3c_2c'])
    conv3c_2c = bias('conv3c_2c', conv3c_2c, _biases['bc3c_2c'])

    conv3c = tf.add(pool3, conv3c_2c)
    conv3c = batch_norm(conv3c, _bnorm['bn3c_2c'], train_phase, scope='bn')
    conv3c = relu('conv3c', conv3c)
    #[path,16,16,512]

    conv3d_2a = cbbr('conv3d_2a', conv3c, _weights['wc3d_2a'], _biases['bc3d_2a'], _bnorm['bn3d_2a'])
    conv3d_2b = cbbr('conv3d_2b', conv3d_2a, _weights['wc3d_2b'], _biases['bc3d_2b'], _bnorm['bn3d_2b'])
    conv3d_2c = conv2d('conv3d_2c', conv3d_2b, _weights['wc3d_2c'])
    conv3d_2c = bias('conv3d_2c', conv3d_2c, _biases['bc3d_2c'])

    conv3d = tf.add(conv3c, conv3d_2c)
    conv3d = batch_norm(conv3d, _bnorm['bn3d_2c'], train_phase, scope='bn')
    conv3d = relu('conv3c', conv3d)
    #[path,16,16,512]
    pool4 = avg_pool('pool4', conv3d, k=2)
    #[path,8,8,512]


    dense1 = tf.reshape(pool4, [-1, _weights['out'].get_shape().as_list()[0]])
    #[patch,hight*width*channel]
    out = tf.add(tf.matmul(dense1, _weights['out']), _biases['bout'])
    return out


pred = res_new(x, weights, biases, bnorm)

#pred = tf.div(pred, tf.reshape(tf.tile(tf.reduce_sum(pred,1),[100]),[batch_size,100]), name='final_norm')


cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

top1=top_k_error(pred,y,1)
top5=top_k_error(pred,y,5)

saver = tf.train.Saver(tf.all_variables())
init = tf.initialize_all_variables()


with tf.Session() as sess:
    sess.run(init)
    step = 1
    train = False
    if train:
        #saver.restore(sess,"resnew-2000")
        #step=2001
        while step  <= training_iters:
            batch = rb.generate_samples(batch_size)
            batch_xs = batch[0]
            batch_ys = batch[1]

            print('Start train step ',str(step))
            sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys, train_phase: True})
            print('Finish step', str(step))
            if step % display_step == 0:

                acc1 = sess.run(top1, feed_dict={x: batch_xs, y: batch_ys, train_phase: False})
                acc5 = sess.run(top5, feed_dict={x: batch_xs, y: batch_ys, train_phase: False})
                loss = sess.run(cost, feed_dict={x: batch_xs, y: batch_ys, train_phase: False})
                print( "Iter " + str(step) + ", Minibatch Loss= " + "{:.6f}".format(loss) + ", Top1= " + "{:.5f}".format(acc1)+ ", Top5= " + "{:.5f}".format(acc5))
            if step % val_step == 0:
                val_batch = rb.generate_val_samples(batch_size)
                val_batch_xs = val_batch[0]
                val_batch_ys = val_batch[1]
                val_error5=sess.run(top5, feed_dict={x: val_batch_xs, y: val_batch_ys, train_phase: False})
                for i in range(3):
                    val_batch = rb.generate_val_samples(batch_size)
                    val_batch_xs = val_batch[0]
                    val_batch_ys = val_batch[1]
                    new_error5 = sess.run(top5, feed_dict={x: val_batch_xs, y: val_batch_ys, train_phase: False})
                    val_error5 = (val_error5*(i+1)+new_error5)/float(i+2)
                print( "Validation Top5= "+ "{:.5f}".format(val_error5))

            if step % save_step == 0:
                saver.save(sess,'resnew',global_step=step)

            step += 1
        print ("Optimization Finished!")

    else:
        evl_size=batch_size
        saver.restore(sess, "resnew-6000")
        while 1==1:
            batch = rb.generate_val_samples(evl_size)
            batch_xs = batch[0]
            batch_ys = batch[1]
            loss = sess.run(cost, feed_dict={x: batch_xs, y: batch_ys, train_phase: False})
            print("Minibatch Loss= " + "{:.6f}".format(loss))
            error1 = sess.run(top1, feed_dict={x: batch_xs, y: batch_ys, train_phase: False})
            error5 = sess.run(top5, feed_dict={x: batch_xs, y: batch_ys, train_phase: False})
            print("Top5 Error= " + "{:.5f}".format(error5) + "   Top1 Error= " + "{:.5f}".format(error1))
