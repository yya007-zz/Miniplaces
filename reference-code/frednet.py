
import fredrb as rb
import tensorflow as tf
from fredpara import *

learning_rate = 0.001
training_iters = 100000
batch_size = 32
display_step = 20
save_step = 2000

n_input = 126*126*3
n_classes = 100

x = tf.placeholder(tf.float32, [batch_size,126,126,3])
xx = tf.placeholder(tf.float32, [batch_size,42,126,3])
xxx = tf.placeholder(tf.float32, [batch_size,42,126,3])
y = tf.placeholder(tf.int32, [batch_size])
keep_prob = tf.placeholder(tf.float32)


def top_k_error(predictions, labels, k):
    in_top1 = tf.to_float(tf.nn.in_top_k(predictions, labels, k))
    num_correct = tf.reduce_sum(in_top1)
    return (batch_size - num_correct) / batch_size

def conv2d(name, l_input, w):
    #tf.nn.conv2d([patch,hight,width,channel], [filter_height, filter_width, in_channels, out_channels], strides=[patch,hight,width,channel])
    return tf.nn.conv2d(l_input, w, strides=[1, 1, 1, 1], padding='SAME', name=name)

def bias(name, l_input,b):
    return tf.nn.bias_add(l_input,b,name=name)

def relu(name, l_input):
    return tf.nn.relu(l_input,name=name)

def max_pool(name, l_input, k=2,s=2):
    # ksize[patch,height,width,channel], strides[patch,height,width,channel]
    return tf.nn.max_pool(l_input, ksize=[1, k, k, 1], strides=[1, s, s, 1], padding='SAME', name=name)

def avg_pool(name, l_input, k,s=1):
    # ksize[patch,height,width,channel], strides[patch,height,width,channel]
    return tf.nn.avg_pool(l_input, ksize=[1, k, k, 1], strides=[1, s, s, 1], padding='SAME', name=name)

def norm(name, l_input, depth_radius=5):
    return tf.nn.lrn(l_input, depth_radius, bias=1.0, alpha=0.0001, beta=0.75, name=name)

def cbrn(name, l_input, w, b):
    return norm(name,relu(name,bias(name,conv2d(name, l_input,w),b)))

def res_net(_X,_XX,_XXX, _weights, _biases):

    conv1 = cbrn('conv1', _X, _weights['wc1'], _biases['bc1'])
    pool1 = max_pool('pool1', conv1, k=3, s=2)
    #[path,63,63,64]

    conv2a_2a = cbrn('conv2a_2a', pool1, _weights['wc2a_2a'], _biases['bc2a_2a'])
    conv2a_2b = cbrn('conv2a_2b', conv2a_2a, _weights['wc2a_2b'], _biases['bc2a_2b'])
    conv2a_2c = conv2d('conv2a_2c', conv2a_2b, _weights['wc2a_2c'])
    conv2a_2c = bias('conv2a_2c', conv2a_2c, _biases['bc2a_2c'])
    conv2a_2c = norm('conv2a_2c', conv2a_2c)

    conv2a_1 = conv2d('conv2a_1', pool1, _weights['wc2a_1'])
    conv2a_1 = bias('conv2a_1', conv2a_1, _biases['bc2a_1'])
    conv2a_1 = norm('conv2a_1', conv2a_1)

    conv2a = relu('conv2a', conv2a_1 + conv2a_2c)
    #[path,63,63,256]

    conv2b_2a = cbrn('conv2b_2a', conv2a, _weights['wc2b_2a'], _biases['bc2b_2a'])
    conv2b_2b = cbrn('conv2b_2b', conv2b_2a, _weights['wc2b_2b'], _biases['bc2b_2b'])
    conv2b_2c = conv2d('conv2b_2c', conv2b_2b, _weights['wc2b_2c'])
    conv2b_2c = bias('conv2b_2c', conv2b_2c, _biases['bc2b_2c'])
    conv2b_2c = norm('conv2b_2c', conv2b_2c)

    conv2b = relu('conv2b', conv2a + conv2b_2c)
    #[path,63,63,256]

    pool2 = avg_pool('pool2', conv2b, k=3,s=2)

    conv2c_2a = cbrn('conv2c_2a', pool2, _weights['wc2c_2a'], _biases['bc2c_2a'])
    conv2c_2b = cbrn('conv2c_2b', conv2c_2a, _weights['wc2c_2b'], _biases['bc2c_2b'])
    conv2c_2c = conv2d('conv2c_2c', conv2c_2b, _weights['wc2c_2c'])
    conv2c_2c = bias('conv2c_2c', conv2c_2c, _biases['bc2c_2c'])
    conv2c_2c = norm('conv2c_2c', conv2c_2c)

    conv2c = relu('conv2c', pool2 + conv2c_2c)
    #[path,32,32,256]

    conv3a_2a = cbrn('conv3a_2a', conv2c, _weights['wc3a_2a'], _biases['bc3a_2a'])
    conv3a_2b = cbrn('conv3a_2b', conv3a_2a, _weights['wc3a_2b'], _biases['bc3a_2b'])
    conv3a_2c = conv2d('conv3a_2c', conv3a_2b, _weights['wc3a_2c'])
    conv3a_2c = bias('conv3a_2c', conv3a_2c, _biases['bc3a_2c'])
    conv3a_2c = norm('conv3a_2c', conv3a_2c)

    conv3a_1 = conv2d('conv3a_1', conv2c, _weights['wc3a_1'])
    conv3a_1 = bias('conv3a_1', conv3a_1, _biases['bc3a_1'])
    conv3a_1 = norm('conv3a_1', conv3a_1)

    conv3a = relu('conv3a', conv3a_1 + conv3a_2c)
    #[path,32,32,512]

    conv3b_2a = cbrn('conv3b_2a', conv3a, _weights['wc3b_2a'], _biases['bc3b_2a'])
    conv3b_2b = cbrn('conv3b_2b', conv3b_2a, _weights['wc3b_2b'], _biases['bc3b_2b'])
    conv3b_2c = conv2d('conv3b_2c', conv3b_2b, _weights['wc3b_2c'])
    conv3b_2c = bias('conv3b_2c', conv3b_2c, _biases['bc3b_2c'])
    conv3b_2c = norm('conv3b_2c', conv3b_2c)

    conv3b = relu('conv3b', conv3a + conv3b_2c)
    #[path,32,32,512]
    pool3 = avg_pool('pool3', conv3b, k=3,s=2)
    #[path,16,16,512]

    conv3c_2a = cbrn('conv3c_2a', pool3, _weights['wc3c_2a'], _biases['bc3c_2a'])
    conv3c_2b = cbrn('conv3c_2b', conv3c_2a, _weights['wc3c_2b'], _biases['bc3c_2b'])
    conv3c_2c = conv2d('conv3c_2c', conv3c_2b, _weights['wc3c_2c'])
    conv3c_2c = bias('conv3c_2c', conv3c_2c, _biases['bc3c_2c'])
    conv3c_2c = norm('conv3c_2c', conv3c_2c)

    conv3c = relu('conv3c', pool3 + conv3c_2c)
    #[path,16,16,512]

    conv3d_2a = cbrn('conv3d_2a', conv3c, _weights['wc3d_2a'], _biases['bc3d_2a'])
    conv3d_2b = cbrn('conv3d_2b', conv3d_2a, _weights['wc3d_2b'], _biases['bc3d_2b'])
    conv3d_2c = conv2d('conv3d_2c', conv3d_2b, _weights['wc3d_2c'])
    conv3d_2c = bias('conv3d_2c', conv3d_2c, _biases['bc3d_2c'])
    conv3d_2c = norm('conv3d_2c', conv3d_2c)

    conv3d = relu('conv3c', conv3c + conv3d_2c)
    #[path,16,16,512]
    pool4 = avg_pool('pool4', conv3d, k=5,s=1)
    #[path,16,16,512]


################################################################################
    up_conv1 = cbrn('up_conv1', _XX, _weights['up_wc1'], _biases['up_bc1'])
    up_pool1 = max_pool('up_pool1', up_conv1, k=3, s=2)
    #[21,63,64]

    up_conv2 = cbrn('up_conv2', up_pool1, _weights['up_wc2'], _biases['up_bc2'])
    up_pool2 = tf.nn.max_pool(up_conv2, ksize=[1, 1, 4, 1], strides=[1, 1, 3, 1], padding='SAME', name='up_pool2')
    #[21,21,96]

    up_conv3 = cbrn('up_conv3', up_pool2, _weights['up_wc3'], _biases['up_bc3'])
    up_pool3 = max_pool('up_pool3', up_conv3, k=3, s=2)
    #[11,11,128]

    up_conv4 = cbrn('up_conv4', up_pool3, _weights['up_wc4'], _biases['up_bc4'])
    up_pool4 = max_pool('up_pool4', up_conv4, k=3, s=2)
    #[6,6,256]

    down_conv1 = cbrn('down_conv1', _XXX, _weights['down_wc1'], _biases['down_bc1'])
    down_pool1 = max_pool('down_pool1', down_conv1, k=3, s=2)

    down_conv2 = cbrn('down_conv2', down_pool1, _weights['down_wc2'], _biases['down_bc2'])
    down_pool2 = tf.nn.max_pool(down_conv2, ksize=[1, 1, 4, 1], strides=[1, 1, 3, 1], padding='SAME', name='down_pool2')

    down_conv3 = cbrn('down_conv3', down_pool2, _weights['down_wc3'], _biases['down_bc3'])
    down_pool3 = max_pool('down_pool3', down_conv3, k=3, s=2)

    down_conv4 = cbrn('down_conv4', down_pool3, _weights['down_wc4'], _biases['down_bc4'])
    down_pool4 = max_pool('down_pool4', down_conv4, k=3, s=2)

    dense1 = tf.reshape(pool4, [-1, 16*16*512])
    dense2 = tf.reshape(up_pool4, [-1, 6*6*256])
    dense3 = tf.reshape(up_pool4, [-1, 6*6*256])
    #[patch,hight*width*channel]
    out1 = tf.matmul(dense1, _weights['out1'])
    out2 = tf.mul( _weights['up_w'],tf.matmul(dense2, _weights['out2']))
    out3 = tf.mul( _weights['down_w'],tf.matmul(dense3, _weights['out3']))
    out = tf.add_n([out1,out2,out3])
    return out


pred = res_net(x,xx,xxx, weights, biases)

#pred = tf.div(pred, tf.reshape(tf.tile(tf.reduce_sum(pred,1),[100]),[batch_size,100]), name='final_norm')


cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,epsilon=0.1).minimize(cost)


# correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
#
# accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
top1=top_k_error(pred,y,1)
#answer=tf.argmax(pred,1)
top5=top_k_error(pred,y,5)

saver = tf.train.Saver(tf.all_variables())
init = tf.initialize_all_variables()


with tf.Session() as sess:
    sess.run(init)
    step = 1
    train = True
    # Keep training until reach max iterations
    if train:
        saver.restore(sess,"frednet-12000")
        step=12001
        while step  <= training_iters:
            batch = rb.generate_triple_samples(batch_size)

            print('Start train step ',str(step))
            sess.run(optimizer, feed_dict={x: batch[0],xx: batch[1],xxx: batch[2], y: batch[3]})
            print('Finish step', str(step))
            if step % display_step == 0:

                acc1 = sess.run(top1, feed_dict={x: batch[0],xx: batch[1],xxx: batch[2], y: batch[3]})
                acc5 = sess.run(top5, feed_dict={x: batch[0],xx: batch[1],xxx: batch[2], y: batch[3]})

                loss = sess.run(cost, feed_dict={x: batch[0],xx: batch[1],xxx: batch[2], y: batch[3]})
                print( "Iter " + str(step) + ", Minibatch Loss= " + "{:.6f}".format(loss) + ", Top1= " + "{:.5f}".format(acc1) + ", Top5= " + "{:.5f}".format(acc5))
            if step % save_step == 0:
                saver.save(sess,'frednet',global_step=step)
            step += 1
        print ("Optimization Finished!")

    else:
        evl_size=32
        while 1==1:
            saver.restore(sess,"frednet-30000")
            batch = rb.generate_val_samples(evl_size)
            # acc1 = sess.run(top1, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
            # acc5 = sess.run(top5, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
            # loss = sess.run(cost, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
            # print("Top5 Error= " + "{:.5f}".format(acc5)+"    Top1 Error= " + "{:.5f}".format(acc1))
            # print("Minibatch Loss= " + "{:.6f}".format(loss))

            #acc5 = sess.run(top5, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
            ans = sess.run(pred, feed_dict={x: batch[0],xx: batch[1],xxx: batch[2], y: batch[3]})
            print(ans)
            print("Top5 Error= " + "{:.5f}".format(acc5))
