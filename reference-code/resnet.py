
import read_batch as rb
import tensorflow as tf
from parameters import *

learning_rate = 0.0005
training_iters = 100000
batch_size = 32
display_step = 20
save_step = 2000

n_input = 128*128*3
n_classes = 100

x = tf.placeholder(tf.float32, [batch_size,128,128,3])
y = tf.placeholder(tf.int32, [batch_size])

def top_k_error(predictions, labels, k):
    in_top1 = tf.to_float(tf.nn.in_top_k(predictions, labels, k))
    num_correct = tf.reduce_sum(in_top1)
    # num=0
    # for img in in_top1:
    #     if img==1:
    #         print(labels[num])
    #     num+=1
    return (batch_size - num_correct) / float(batch_size)

def conv2d(name, l_input, w):
    #tf.nn.conv2d([patch,hight,width,channel], [filter_height, filter_width, in_channels, out_channels], strides=[patch,hight,width,channel])
    return tf.nn.conv2d(l_input, w, strides=[1, 1, 1, 1], padding='SAME', name=name)

def bias(name, l_input,b):
    return tf.nn.bias_add(l_input,b,
name=name)

def relu(name, l_input):
    return tf.nn.relu(l_input,name=name)

def max_pool(name, l_input, k):
    # ksize[patch,height,width,channel], strides[patch,height,width,channel]
    return tf.nn.max_pool(l_input, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME', name=name)

def avg_pool(name, l_input, k):
    # ksize[patch,height,width,channel], strides[patch,height,width,channel]
    return tf.nn.avg_pool(l_input, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME', name=name)

def norm(name, l_input, depth_radius=5):
    return tf.nn.lrn(l_input, depth_radius, bias=1.0, alpha=0.0001, beta=0.75, name=name)

def cbrn(name, l_input, w, b):
    return norm(name,relu(name,bias(name,conv2d(name, l_input,w),b)))

def res_net(_X, _weights, _biases):

    _X = tf.reshape(_X, shape=[-1, 128, 128, 3])

    conv1 = cbrn('conv1', _X, _weights['wc1'], _biases['bc1'])
    pool1 = max_pool('pool1', conv1, k=2)
    #[path,64,64,64]

    conv2a_2a = cbrn('conv2a_2a', pool1, _weights['wc2a_2a'], _biases['bc2a_2a'])
    conv2a_2b = cbrn('conv2a_2b', conv2a_2a, _weights['wc2a_2b'], _biases['bc2a_2b'])
    conv2a_2c = conv2d('conv2a_2c', conv2a_2b, _weights['wc2a_2c'])
    conv2a_2c = bias('conv2a_2c', conv2a_2c, _biases['bc2a_2c'])
    conv2a_2c = norm('conv2a_2c', conv2a_2c)

    conv2a_1 = conv2d('conv2a_1', pool1, _weights['wc2a_1'])
    conv2a_1 = bias('conv2a_1', conv2a_1, _biases['bc2a_1'])
    conv2a_1 = norm('conv2a_1', conv2a_1)

    conv2a = relu('conv2a', conv2a_1 + conv2a_2c)
    #[path,32,32,256]

    conv2b_2a = cbrn('conv2b_2a', conv2a, _weights['wc2b_2a'], _biases['bc2b_2a'])
    conv2b_2b = cbrn('conv2b_2b', conv2b_2a, _weights['wc2b_2b'], _biases['bc2b_2b'])
    conv2b_2c = conv2d('conv2b_2c', conv2b_2b, _weights['wc2b_2c'])
    conv2b_2c = bias('conv2b_2c', conv2b_2c, _biases['bc2b_2c'])
    conv2b_2c = norm('conv2b_2c', conv2b_2c)

    conv2b = relu('conv2b', conv2a + conv2b_2c)
    #[path,32,32,256]

    pool2 = avg_pool('pool2', conv2b, k=2)

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
    pool3 = avg_pool('pool3', conv3b, k=2)
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
    pool4 = avg_pool('pool4', conv3d, k=2)
    #[path,8,8,512]


    dense1 = tf.reshape(pool4, [-1, _weights['out'].get_shape().as_list()[0]])
    #[patch,hight*width*channel]
    out = tf.matmul(dense1, _weights['out'])
    return out


pred = res_net(x, weights, biases)

#pred = tf.div(pred, tf.reshape(tf.tile(tf.reduce_sum(pred,1),[100]),[batch_size,100]), name='final_norm')


cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,epsilon=0.1).minimize(cost)


# correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
#
# accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
top1=top_k_error(pred,y,1)
answer=tf.argmax(pred,1)
top5=top_k_error(pred,y,5)

saver = tf.train.Saver(tf.all_variables())
init = tf.initialize_all_variables()


with tf.Session() as sess:
    sess.run(init)
    step = 1
    train = True
    # Keep training until reach max iterations
    if train:
        saver.restore(sess,"resnet-27000")
        step=27001
        while step  <= training_iters:
            batch = rb.generate_samples(batch_size)
            batch_xs = batch[0]
            batch_ys = batch[1]
            # if step==300:
            #     learning_rate = 0.0005
            # if step==800:
            #     learning_rate = 0.0002
            # if step==1200:
            #     learning_rate = 0.00002
            print('Start train step ',str(step))
            sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})
            print('Finish step', str(step))
            if step % display_step == 0:

                acc1 = sess.run(top1, feed_dict={x: batch_xs, y: batch_ys})
                acc5 = sess.run(top5, feed_dict={x: batch_xs, y: batch_ys})

                loss = sess.run(cost, feed_dict={x: batch_xs, y: batch_ys})
                print( "Iter " + str(step) + ", Minibatch Loss= " + "{:.6f}".format(loss) + ", Top1= " + "{:.5f}".format(acc1)+ ", Top5= " + "{:.5f}".format(acc5))
            if step % save_step == 0:
                saver.save(sess,'resnet',global_step=step)
            step += 1
        print ("Optimization Finished!")

    else:
        evl_size=32
        while 1==1:
            saver.restore(sess,"resnet-30000")
            batch = rb.generate_val_samples(evl_size)
            batch_xs = batch[0]
            batch_ys = batch[1]
            # acc1 = sess.run(top1, feed_dict={x: batch_xs, y: batch_ys})
            # acc5 = sess.run(top5, feed_dict={x: batch_xs, y: batch_ys})
            # loss = sess.run(cost, feed_dict={x: batch_xs, y: batch_ys})
            # print("Top5 Error= " + "{:.5f}".format(acc5)+"    Top1 Error= " + "{:.5f}".format(acc1))
            # print("Minibatch Loss= " + "{:.6f}".format(loss))

            #acc5 = sess.run(top5, feed_dict={x: batch_xs, y: batch_ys})
            ans = sess.run(pred, feed_dict={x: batch_xs, y: batch_ys})
            print(ans)
            print("Top5 Error= " + "{:.5f}".format(acc5))
