import os, datetime
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from DataLoader import *
from DataLoaderNoise import DataLoaderDiskRandomize
from architect import *
from save import *

# Training Parameters
learning_rate = 0.00001
training_iters = 25000
batch_size = 32
step_display = 100
step_save = 1000
exp_name = 'exp4'
num = ''

train = True
validation = False
test = False

path_save = './save-'+exp_name+'/'
start_from=''
if len(num)>0:
    start_from = path_save+'-'+num


load_size = 256
fine_size = 224
c = 3
data_mean = np.asarray([0.45834960097,0.44674252445,0.41352266842])
dropout = 0.5 # Dropout, probability to keep units

# Construct dataloader
opt_data_train = {
    #'data_h5': 'miniplaces_256_train.h5',
    'data_root': '../../../data/images/',   # MODIFY PATH ACCORDINGLY
    'data_list': '../../../data/train.txt', # MODIFY PATH ACCORDINGLY
    'load_size': load_size,
    'fine_size': fine_size,
    'data_mean': data_mean,
    'randomize': True,
    'perm' : True
    }

opt_data_val = {
    #'data_h5': 'miniplaces_256_val.h5',
    'data_root': '../../../data/images/',   # MODIFY PATH ACCORDINGLY
    'data_list': '../../../data/val.txt',   # MODIFY PATH ACCORDINGLY
    'load_size': load_size,
    'fine_size': fine_size,
    'data_mean': data_mean,
    'randomize': False,
    'perm' : False
    }

opt_data_test = {
    #'data_h5': 'miniplaces_256_val.h5',
    'data_root': '../../../data/images/',   # MODIFY PATH ACCORDINGLY
    'data_list': '../../../data/test.txt',   # MODIFY PATH ACCORDINGLY
    'load_size': load_size,
    'fine_size': fine_size,
    'data_mean': data_mean,
    'randomize': False,
    'perm' : False
    }

loader_train = DataLoaderDiskRandomize(**opt_data_train)
loader_val = DataLoaderDisk(**opt_data_val)
loader_test = DataLoaderDisk(**opt_data_test)

print ("finish loading data")
# tf Graph input
x = tf.placeholder(tf.float32, [None, fine_size, fine_size, c])
y = tf.placeholder(tf.int64, None)
keep_dropout = tf.placeholder(tf.float32)
train_phase = tf.placeholder(tf.bool)

# # Construct model
model = vgg_model(x, y, keep_dropout, train_phase)

# Define loss and optimizer
logits= model.logits
loss = model.loss
train_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# Evaluate model
accuracy1 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(logits, y, 1), tf.float32))
accuracy5 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(logits, y, 5), tf.float32))

# define initialization
init = tf.global_variables_initializer()

# define saver
saver = tf.train.Saver()

# define summary writer
#writer = tf.train.SummaryWriter('.', graph=tf.get_default_graph())

# Launch the graph
with tf.Session() as sess:
    # Initialization
    if len(start_from)>1:
        saver.restore(sess, start_from)
    else:
        sess.run(init)

    def validation():
        t=time.time()
        # Evaluate on the whole validation set
        print('Evaluation on the whole validation set...')
        num_batch = loader_val.size()//batch_size+1
        acc1_total = 0.
        acc5_total = 0.
        loader_val.reset()
        for i in range(num_batch):
            images_batch, labels_batch = loader_val.next_batch(batch_size)    
            acc1, acc5 = sess.run([accuracy1, accuracy5], feed_dict={x: images_batch, y: labels_batch, keep_dropout: 1., train_phase: False})
            acc1_total += acc1
            acc5_total += acc5
        acc1_total /= num_batch
        acc5_total /= num_batch
        t=int(time.time()-t)
        print('used'+str(t)+'s to validate')
        print('Evaluation Finished! Accuracy Top1 = ' + "{:.4f}".format(acc1_total) + ", Top5 = " + "{:.4f}".format(acc5_total))
        return acc1_total,acc5_total
    
    step = 0

    if train:
        train_accs=[]
        val_accs=[]
        while step < training_iters:
            # Load a batch of training data
            images_batch, labels_batch = loader_train.next_batch(batch_size)
            
            if step % step_display == 0:
                print('[%s]:' %(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

                # Calculate batch loss and accuracy on training set
                l, lo, acc1, acc5 = sess.run([loss, logits, accuracy1, accuracy5], feed_dict={x: images_batch, y: labels_batch, keep_dropout: 1., train_phase: False}) 
                print("-Iter " + str(step) + ", Training Loss= " + \
                      "{:.6f}".format(l) + ", Accuracy Top1 = " + \
                      "{:.4f}".format(acc1) + ", Top5 = " + \
                      "{:.4f}".format(acc5))

                print(lo[1])

                # acc1, acc5=validation()
                # val_accs.append(acc5)

                # fig = plt.figure()
                # a=np.arange(1,len(val_accs)+1,1)
                # plt.plot(a,train_accs,'-',label='Training')
                # plt.plot(a,val_accs,'-',label='Validation')
                # plt.xlabel("Iteration")
                # plt.ylabel("Accuracy")
                # plt.legend()
                # fig.savefig("./fig/pic_"+str(exp_name)+".png")   # save the figure to file
                # plt.close(fig)
                # print "finish saving figure to view"
            
            # Run optimization op (backprop)
            sess.run(train_optimizer, feed_dict={x: images_batch, y: labels_batch, keep_dropout: dropout, train_phase: True})
            
            step += 1
            
            # Save model
            if step % step_save == 0:
                saver.save(sess, path_save, global_step=step)
                print("Model saved at Iter %d !" %(step))
        print("Optimization Finished!")

    if validation:
        validation()
    
    if test:
        # Predict on the test set
        print('Evaluation on the test set...')
        num_batch = loader_test.size()//batch_size+1
        loader_test.reset()
        result=[]
        for i in range(num_batch):
            images_batch, labels_batch = loader_test.next_batch(batch_size) 
            l = sess.run([logits], feed_dict={x: images_batch, y: labels_batch, keep_dropout: 1., train_phase: False})
            l = np.array(l)
            l = l.reshape(l.shape[1:])
            print l.shape
            for ind in range(l.shape[0]):
                top5 = np.argsort(l[ind])[-5:][::-1]
                result.append(top5)
        result=np.array(result)
        result=result[:10000,:]
        save(result, "./fig/"+exp_name+str(num))