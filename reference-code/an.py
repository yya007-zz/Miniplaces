
import read_batch as rb
import tensorflow as tf

learning_rate = 0.001
training_iters = 20000
batch_size = 64
display_step = 20
save_step = 500

n_input = 128*128*3 
n_classes = 10 
dropout = 0.5 

x = tf.placeholder(tf.float32, [batch_size,128,128,3])
y = tf.placeholder(tf.float32, [batch_size, n_classes])
keep_prob = tf.placeholder(tf.float32)

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

def conv2d(name, l_input, w, b):
    #tf.nn.conv2d([patch,hight,width,channel], [filter_height, filter_width, in_channels, out_channels], strides=[patch,hight,width,channel])
    return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(l_input, w, strides=[1, 1, 1, 1], padding='SAME'),b), name=name)


def max_pool(name, l_input, k):
    # ksize[patch,height,width,channel], strides[patch,height,width,channel]
    return tf.nn.max_pool(l_input, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME', name=name)


def norm(name, l_input, lsize=4):
    return tf.nn.lrn(l_input, lsize, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name=name)


def alex_net(_X, _weights, _biases, _dropout):
    _X = tf.reshape(_X, shape=[-1, 128, 128, 3])
    conv1 = conv2d('conv1', _X, _weights['wc1'], _biases['bc1'])
    pool1 = max_pool('pool1', conv1, k=2)
    norm1 = norm('norm1', pool1, lsize=4)
    norm1 = tf.nn.dropout(norm1, _dropout)
    conv2 = conv2d('conv2', norm1, _weights['wc2'], _biases['bc2'])
    pool2 = max_pool('pool2', conv2, k=2)
    norm2 = norm('norm2', pool2, lsize=4)
    norm2 = tf.nn.dropout(norm2, _dropout)
    conv3 = conv2d('conv3', norm2, _weights['wc3'], _biases['bc3'])
    pool3 = max_pool('pool3', conv3, k=2)
    norm3 = norm('norm3', pool3, lsize=4)
    norm3 = tf.nn.dropout(norm3, _dropout)
    conv4 = conv2d('conv4', norm3, _weights['wc4'], _biases['bc4'])
    conv4_2 = conv2d('conv4_2', conv4, _weights['wc4_2'], _biases['bc4_2'])
    conv4_3 = conv2d('conv4_3', conv4_2, _weights['wc4_3'], _biases['bc4_3'])
    pool4 = max_pool('pool4', conv4_3, k=2)
    norm4 = norm('norm4', pool4, lsize=4)
    norm4 = tf.nn.dropout(norm4, _dropout)
    # conv5 = conv2d('conv5', norm4, _weights['wc5'], _biases['bc5'])
    # pool5 = max_pool('pool5', conv5, k=2)
    # norm5 = norm('norm5', pool5, lsize=4)
    # norm5 = tf.nn.dropout(norm5, _dropout)
    dense1 = tf.reshape(norm4, [-1, _weights['wd1'].get_shape().as_list()[0]])
    dense1 = tf.nn.relu(tf.matmul(dense1, _weights['wd1']) + _biases['bd1'], name='fc1')
    dense2 = tf.nn.relu(tf.matmul(dense1, _weights['wd2']) + _biases['bd2'], name='fc2')
    out = tf.matmul(dense2, _weights['out']) + _biases['out']
    return out

weights = {
    'wc1': tf.Variable(tf.random_normal([7, 7, 3, 96])),
    'wc2': tf.Variable(tf.random_normal([5, 5, 96, 128])),
    'wc3': tf.Variable(tf.random_normal([3, 3, 128, 256])),
    'wc4': tf.Variable(tf.random_normal([3, 3, 256, 512])),
    'wc4_2': tf.Variable(tf.random_normal([3, 3, 512, 512])),
    'wc4_3': tf.Variable(tf.random_normal([3, 3, 512, 256])),
    # 'wc5': tf.Variable(tf.random_normal([3, 3, 512, 256])),
    'wd1': tf.Variable(tf.random_normal([8*8*256, 1024])),
    'wd2': tf.Variable(tf.random_normal([1024, 1024])),
    'out': tf.Variable(tf.random_normal([1024, n_classes]))
}
biases = {
    'bc1': tf.Variable(tf.random_normal([96])),
    'bc2': tf.Variable(tf.random_normal([128])),
    'bc3': tf.Variable(tf.random_normal([256])),
    'bc4': tf.Variable(tf.random_normal([512])),
    'bc4_2': tf.Variable(tf.random_normal([512])),
    'bc4_3': tf.Variable(tf.random_normal([256])),
    # 'bc5': tf.Variable(tf.random_normal([256])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'bd2': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

pred = alex_net(x, weights, biases, keep_prob)


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
        # saver.restore(sess,"resnet-27000")
        # step=27001
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
            sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys, keep_prob: dropout})
            print('Finish step', str(step))
            if step % display_step == 0:

                acc1 = sess.run(top1, feed_dict={x: batch_xs, y: batch_ys, keep_prob:1.})
                acc5 = sess.run(top5, feed_dict={x: batch_xs, y: batch_ys, keep_prob:1.})

                loss = sess.run(cost, feed_dict={x: batch_xs, y: batch_ys, keep_prob:1.})
                print( "Iter " + str(step) + ", Minibatch Loss= " + "{:.6f}".format(loss) + ", Top1= " + "{:.5f}".format(acc1)+ ", Top5= " + "{:.5f}".format(acc5))
            if step % save_step == 0:
                saver.save(sess,'resnet',global_step=step)
            step += 1
        print ("Optimization Finished!")

    else:
        evl_size=32
        while 1==1:
            #saver.restore(sess,"resnet-30000")
            batch = rb.generate_val_samples(evl_size)
            batch_xs = batch[0]
            batch_ys = batch[1]
            # acc1 = sess.run(top1, feed_dict={x: batch_xs, y: batch_ys})
            # acc5 = sess.run(top5, feed_dict={x: batch_xs, y: batch_ys})
            # loss = sess.run(cost, feed_dict={x: batch_xs, y: batch_ys})
            # print("Top5 Error= " + "{:.5f}".format(acc5)+"    Top1 Error= " + "{:.5f}".format(acc1))
            # print("Minibatch Loss= " + "{:.6f}".format(loss))

            #acc5 = sess.run(top5, feed_dict={x: batch_xs, y: batch_ys})
            ans = sess.run(pred, feed_dict={x: batch_xs, y: batch_ys, keep_prob:1.})
            print(ans)
            print("Top5 Error= " + "{:.5f}".format(acc5))