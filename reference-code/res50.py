import bn_read_2 as rb
import tensorflow as tf
from para50 import *
import numpy as np

train = True
test = False
visualize = False

learning_rate = 0.001
training_iters = 50001
batch_size = 32
if test:
    batch_size = 18
display_step = 20
save_step = 5000
val_step = 20

train_top5=[]
val_top5=[]

n_input = 126*126*3
n_classes = 100

x = tf.placeholder(tf.float32, [batch_size,126,126,3])
y = tf.placeholder(tf.int32, [batch_size])
train_phase = tf.placeholder(tf.bool, name='train_phase')
#To tell batch_norm whether to update global mean and var

def put_kernels_on_grid (kernel, grid_Y, grid_X, pad=1):
    '''Visualize conv. features as an image (mostly for the 1st layer).
    Place kernel into a grid, with some paddings between adjacent filters.
    Args:
      kernel:            tensor of shape [Y, X, NumChannels, NumKernels]
      (grid_Y, grid_X):  shape of the grid. Require: NumKernels == grid_Y * grid_X
                           User is responsible of how to break into two multiples.
      pad:               number of black pixels around each filter (between them)
    Return:
      Tensor of shape [(Y+pad)*grid_Y, (X+pad)*grid_X, NumChannels, 1].
    '''
    x1 = tf.pad(kernel, tf.constant( [[pad,0],[pad,0],[0,0],[0,0]] ))
    Y = kernel.get_shape()[0] + pad
    X = kernel.get_shape()[1] + pad
    x2 = tf.transpose(x1, (3, 0, 1, 2))
    x3 = tf.reshape(x2, tf.pack([grid_X, Y * grid_Y, X, 3]))
    x4 = tf.transpose(x3, (0, 2, 1, 3))
    x5 = tf.reshape(x4, tf.pack([1, X * grid_X, Y * grid_Y, 3]))
    x6 = tf.transpose(x5, (2, 1, 3, 0))
    x7 = tf.transpose(x6, (3, 0, 1, 2))
    x_min = tf.reduce_min(x7)
    x_max = tf.reduce_max(x7)
    x8 = (x7 - x_min) / (x_max - x_min)
    return x8

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

def max5(pr):
    pr = np.array(pr)
    ls = []
    for i in range(5):
        ag = np.argmax(pr)
        ls.append(ag)
        pr[ag] = -100
    return ls

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
def avg_pool(name, l_input, k, s=1):
    # ksize[patch,height,width,channel], strides[patch,height,width,channel]
    return tf.nn.avg_pool(l_input, ksize=[1, k, k, 1], strides=[1, s, s, 1], padding='SAME', name=name)
def cbbr(name, l_input, w, b, bn):
    return relu(name,bias(name,batch_norm(conv2d(name, l_input,w), bn, train_phase, scope='bn'),b))
def cpbbr(name, l_input, w, b, bn):
    return relu(name,bias(name,batch_norm(conv2dp(name, l_input,w), bn, train_phase, scope='bn'),b))
def cbb(name, l_input, w, b, bn):
    return bias(name,batch_norm(conv2d(name, l_input,w), bn, train_phase, scope='bn'),b)
def cpbb(name, l_input, w, b, bn):
    return bias(name,batch_norm(conv2dp(name, l_input,w), bn, train_phase, scope='bn'),b)

###############################################################################

def res_new(_X, _weights, _biases, _bnorm):

    conv1 = cpbbr('conv1', _X, _weights['wc1'], _biases['bc1'], _bnorm['bn1'])
    #[path,63,63,64]

    conv2a_2a = cbbr('conv2a_2a', conv1, _weights['wc2a_2a'], _biases['bc2a_2a'], _bnorm['bn2a_2a'])
    conv2a_2b = cbbr('conv2a_2b', conv2a_2a, _weights['wc2a_2b'], _biases['bc2a_2b'], _bnorm['bn2a_2b'])
    conv2a_2c = cbb('conv2a_2c', conv2a_2b, _weights['wc2a_2c'], _biases['bc2a_2c'], _bnorm['bn2a_2c'])

    conv2a_1 = cbb('conv2a_1', conv1, _weights['wc2a_1'], _biases['bc2a_1'], _bnorm['bn2a_1'])

    conv2a = relu('conv2a', tf.add( conv2a_1 , conv2a_2c))
    #[path,63,63,256]


    conv2b_2a = cbbr('conv2b_2a', conv2a, _weights['wc2b_2a'], _biases['bc2b_2a'], _bnorm['bn2b_2a'])
    conv2b_2b = cbbr('conv2b_2b', conv2b_2a, _weights['wc2b_2b'], _biases['bc2b_2b'], _bnorm['bn2b_2b'])
    conv2b_2c = cbb('conv2a_2c', conv2b_2b, _weights['wc2b_2c'], _biases['bc2b_2c'], _bnorm['bn2b_2c'])

    conv2b = relu('conv2b', tf.add( conv2a , conv2b_2c ))
    #[path,63,63,256]

    conv2c_2a = cbbr('conv2c_2a', conv2b, _weights['wc2c_2a'], _biases['bc2c_2a'], _bnorm['bn2c_2a'])
    conv2c_2b = cbbr('conv2c_2b', conv2c_2a, _weights['wc2c_2b'], _biases['bc2c_2b'], _bnorm['bn2c_2b'])
    conv2c_2c = cbb('conv2c_2c', conv2c_2b, _weights['wc2c_2c'], _biases['bc2c_2c'], _bnorm['bn2c_2c'])

    conv2c = relu('conv2c', tf.add( conv2b , conv2c_2c ))
    #[path,63,63,256]

###################################################################################
    conv3a_2a = cpbbr('conv3a_2a', conv2c, _weights['wc3a_2a'], _biases['bc3a_2a'], _bnorm['bn3a_2a'])
    conv3a_2b = cbbr('conv3a_2b', conv3a_2a, _weights['wc3a_2b'], _biases['bc3a_2b'], _bnorm['bn3a_2b'])
    conv3a_2c = cbb('conv3a_2c', conv3a_2b, _weights['wc3a_2c'], _biases['bc3a_2c'], _bnorm['bn3a_2c'])
    #[path,32,32,512]

    conv3a_1 = cpbb('conv3a_1', conv2c, _weights['wc3a_1'], _biases['bc3a_1'], _bnorm['bn3a_1'])

    conv3a = relu('conv3a', tf.add( conv3a_1 , conv3a_2c ))
    #[path,32,32,512]

    conv3b_2a = cbbr('conv3b_2a', conv3a, _weights['wc3b_2a'], _biases['bc3b_2a'], _bnorm['bn3b_2a'])
    conv3b_2b = cbbr('conv3b_2b', conv3b_2a, _weights['wc3b_2b'], _biases['bc3b_2b'], _bnorm['bn3b_2b'])
    conv3b_2c = cbb('conv3b_2c', conv3b_2b, _weights['wc3b_2c'], _biases['bc3b_2c'], _bnorm['bn3b_2c'])

    conv3b = relu('conv3b', tf.add( conv3a , conv3b_2c ))
    #[path,32,32,512]

    conv3c_2a = cbbr('conv3c_2a', conv3b, _weights['wc3c_2a'], _biases['bc3c_2a'], _bnorm['bn3c_2a'])
    conv3c_2b = cbbr('conv3c_2b', conv3c_2a, _weights['wc3c_2b'], _biases['bc3c_2b'], _bnorm['bn3c_2b'])
    conv3c_2c = cbb('conv3c_2c', conv3c_2b, _weights['wc3c_2c'], _biases['bc3c_2c'], _bnorm['bn3c_2c'])

    conv3c = relu('conv3c', tf.add (conv3b , conv3c_2c ))
    #[path,32,32,512]

    conv3d_2a = cbbr('conv3d_2a', conv3c, _weights['wc3d_2a'], _biases['bc3d_2a'], _bnorm['bn3d_2a'])
    conv3d_2b = cbbr('conv3d_2b', conv3d_2a, _weights['wc3d_2b'], _biases['bc3d_2b'], _bnorm['bn3d_2b'])
    conv3d_2c = cbb('conv3d_2c', conv3d_2b, _weights['wc3d_2c'], _biases['bc3d_2c'], _bnorm['bn3d_2c'])

    conv3d = relu('conv3d', tf.add (conv3c , conv3d_2c ))
    #[path,32,32,512]

#################################################################
    conv4a_2a = cpbbr('conv4a_2a', conv3d, _weights['wc4a_2a'], _biases['bc4a_2a'], _bnorm['bn4a_2a'])
    conv4a_2b = cbbr('conv4a_2b', conv4a_2a, _weights['wc4a_2b'], _biases['bc4a_2b'], _bnorm['bn4a_2b'])
    conv4a_2c = cbb('conv4a_2c', conv4a_2b, _weights['wc4a_2c'], _biases['bc4a_2c'], _bnorm['bn4a_2c'])
    #[path,16,16,1024]

    conv4a_1 = cpbb('conv4a_1', conv3d, _weights['wc4a_1'], _biases['bc4a_1'], _bnorm['bn4a_1'])

    conv4a = relu('conv4a', tf.add( conv4a_1 , conv4a_2c ))
    #[path,16,16,1024]

    conv4b_2a = cbbr('conv4b_2a', conv4a, _weights['wc4b_2a'], _biases['bc4b_2a'], _bnorm['bn4b_2a'])
    conv4b_2b = cbbr('conv4b_2b', conv4b_2a, _weights['wc4b_2b'], _biases['bc4b_2b'], _bnorm['bn4b_2b'])
    conv4b_2c = cbb('conv4b_2c', conv4b_2b, _weights['wc4b_2c'], _biases['bc4b_2c'], _bnorm['bn4b_2c'])

    conv4b = relu('conv4b', tf.add( conv4a , conv4b_2c ))
    #[path,16,16,1024]

    conv4c_2a = cbbr('conv4c_2a', conv4b, _weights['wc4c_2a'], _biases['bc4c_2a'], _bnorm['bn4c_2a'])
    conv4c_2b = cbbr('conv4c_2b', conv4c_2a, _weights['wc4c_2b'], _biases['bc4c_2b'], _bnorm['bn4c_2b'])
    conv4c_2c = cbb('conv4c_2c', conv4c_2b, _weights['wc4c_2c'], _biases['bc4c_2c'], _bnorm['bn4c_2c'])

    conv4c = relu('conv4c', tf.add (conv4b , conv4c_2c ))
    #[path,16,16,1024]

    conv4d_2a = cbbr('conv4d_2a', conv4c, _weights['wc4d_2a'], _biases['bc4d_2a'], _bnorm['bn4d_2a'])
    conv4d_2b = cbbr('conv4d_2b', conv4d_2a, _weights['wc4d_2b'], _biases['bc4d_2b'], _bnorm['bn4d_2b'])
    conv4d_2c = cbb('conv4d_2c', conv4d_2b, _weights['wc4d_2c'], _biases['bc4d_2c'], _bnorm['bn4d_2c'])

    conv4d = relu('conv4d', tf.add (conv4c , conv4d_2c ))
    #[path,16,16,1024]

    conv4e_2a = cbbr('conv4e_2a', conv4d, _weights['wc4e_2a'], _biases['bc4e_2a'], _bnorm['bn4e_2a'])
    conv4e_2b = cbbr('conv4e_2b', conv4e_2a, _weights['wc4e_2b'], _biases['bc4e_2b'], _bnorm['bn4e_2b'])
    conv4e_2c = cbb('conv4e_2c', conv4e_2b, _weights['wc4e_2c'], _biases['bc4e_2c'], _bnorm['bn4e_2c'])

    conv4e = relu('conv4e', tf.add (conv4d , conv4e_2c ))
    #[path,16,16,1024]

    conv4f_2a = cbbr('conv4f_2a', conv4e, _weights['wc4f_2a'], _biases['bc4f_2a'], _bnorm['bn4f_2a'])
    conv4f_2b = cbbr('conv4f_2b', conv4f_2a, _weights['wc4f_2b'], _biases['bc4f_2b'], _bnorm['bn4f_2b'])
    conv4f_2c = cbb('conv4f_2c', conv4f_2b, _weights['wc4f_2c'], _biases['bc4f_2c'], _bnorm['bn4f_2c'])

    conv4f = relu('conv4f', tf.add (conv4e , conv4f_2c ))
    #[path,16,16,1024]
###################################################################
    conv5a_2a = cpbbr('conv5a_2a', conv4f, _weights['wc5a_2a'], _biases['bc5a_2a'], _bnorm['bn5a_2a'])
    conv5a_2b = cbbr('conv5a_2b', conv5a_2a, _weights['wc5a_2b'], _biases['bc5a_2b'], _bnorm['bn5a_2b'])
    conv5a_2c = cbb('conv5a_2c', conv5a_2b, _weights['wc5a_2c'], _biases['bc5a_2c'], _bnorm['bn5a_2c'])
    #[path,8,8,2048]

    conv5a_1 = cpbb('conv5a_1', conv4f, _weights['wc5a_1'], _biases['bc5a_1'], _bnorm['bn5a_1'])

    conv5a = relu('conv5a', tf.add( conv5a_1 , conv5a_2c ))
    #[path,8,8,2048]

    conv5b_2a = cbbr('conv5b_2a', conv5a, _weights['wc5b_2a'], _biases['bc5b_2a'], _bnorm['bn5b_2a'])
    conv5b_2b = cbbr('conv5b_2b', conv5b_2a, _weights['wc5b_2b'], _biases['bc5b_2b'], _bnorm['bn5b_2b'])
    conv5b_2c = cbb('conv5b_2c', conv5b_2b, _weights['wc5b_2c'], _biases['bc5b_2c'], _bnorm['bn5b_2c'])

    conv5b = relu('conv5b', tf.add( conv5a , conv5b_2c ))
    #[path,8,8,2048]

    conv5c_2a = cbbr('conv5c_2a', conv5b, _weights['wc5c_2a'], _biases['bc5c_2a'], _bnorm['bn5c_2a'])
    conv5c_2b = cbbr('conv5c_2b', conv5c_2a, _weights['wc5c_2b'], _biases['bc5c_2b'], _bnorm['bn5c_2b'])
    conv5c_2c = cbb('conv5c_2c', conv5c_2b, _weights['wc5c_2c'], _biases['bc5c_2c'], _bnorm['bn5c_2c'])

    conv5c = relu('conv5c', tf.add (conv5b , conv5c_2c ))
    #[path,8,8,2048]

    pool5 = avg_pool('pool5', conv5c, k=5, s=1)
    #[path,8,8,2048]


    dense1 = tf.reshape(pool5, [-1, _weights['out'].get_shape().as_list()[0]])
    #[patch,hight*width*channel]
    out = tf.matmul(dense1, _weights['out'])
    return out

########################################################################

pred = res_new(x, weights, biases, bnorm)
tst = tf.reduce_mean(pred,0)


cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

top1=top_k_error(pred,y,1)
top5=top_k_error(pred,y,5)

saver = tf.train.Saver(tf.all_variables())
init = tf.initialize_all_variables()

###############################################################################

with tf.Session() as sess:
    sess.run(init)
    step = 1
    if train:
        #saver.restore(sess,"res50-15000")
        #step=15001
        fo = open("train_res50.txt", "w+")
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
                fo.write(str(acc5)+" ")
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
                fo.write(str(val_error5)+"\n")
            if step % save_step == 0:
                saver.save(sess,'res50',global_step=step)
            step += 1
        fo.close()
        print ("Optimization Finished!")

    elif not test and not visualize:
        evl_size=batch_size
        saver.restore(sess, "ban_res2-15000")
        while 1==1:
            batch = rb.generate_val_samples(evl_size)
            batch_xs = batch[0]
            batch_ys = batch[1]
            loss = sess.run(cost, feed_dict={x: batch_xs, y: batch_ys, train_phase: False})
            print("Minibatch Loss= " + "{:.6f}".format(loss))
            error1 = sess.run(top1, feed_dict={x: batch_xs, y: batch_ys, train_phase: False})
            error5 = sess.run(top5, feed_dict={x: batch_xs, y: batch_ys, train_phase: False})
            print("Top5 Error= " + "{:.5f}".format(error5) + "   Top1 Error= " + "{:.5f}".format(error1))

    elif test:
        saver.restore(sess,"ban_res2-15000")
        fake_label=np.zeros([batch_size],dtype=np.int32)
        fo = open("result.txt", "w+")
        for i in range(10000):
            test_sample = rb.generate_test_sample(i)
            name = test_sample[0]
            batch = test_sample[1]
            ans = sess.run(tst, feed_dict={x: batch, y: fake_label, train_phase: False})
            top5 = max5(ans)
            for i in top5:
                name += " " + str(i)
            fo.write(name+"\n")
            print (name)
        fo.close()
    if not train and not test and visualize:
        # Before run the porgram, please run this command:
        # tensorboard --logdir=run1:/tmp/tensorflow/ --port 6006
        saver.restore(sess,"ban_res2-15000")
        kernels=weights['wc1']
        grid_x = grid_y = 8   # to get a square grid for 64 conv1 features
        print('start visualization')
        grid = put_kernels_on_grid (kernels, grid_y, grid_x)
        writer = tf.train.SummaryWriter("/tmp/tensorflow", graph=tf.get_default_graph())
        img = tf.image_summary('conv1/features', grid, max_images=1)
        a = sess.run(img)
        writer.add_summary(a,1)
        print('Visualize successful')
