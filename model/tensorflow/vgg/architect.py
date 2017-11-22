import os, datetime
import numpy as np
import tensorflow as tf
import CNNModels

class vgg_model:
    def __init__(self, x, y, keep_dropout, train_phase):
        self.logits=tf.nn.softmax(VGG(x, keep_dropout, train_phase))
        self.loss=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=self.logits))

def VGG(x, keep_dropout, train_phase,num_classes=100):
    return CNNModels.VGG16(x, keep_dropout, train_phase,num_classes=num_classes)
 