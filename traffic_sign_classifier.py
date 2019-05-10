# -*- coding: utf-8 -*-
"""Traffic_Sign_Classifier.ipynb
"""

# get data from udacity
import requests, zipfile, io, pickle

r = requests.get("https://d17h27t6h515a5.cloudfront.net/topher/2017/February/5898cd6f_traffic-signs-data/traffic-signs-data.zip")
z = zipfile.ZipFile(io.BytesIO(r.content))

train = pickle.load(z.open('train.p'))
test = pickle.load(z.open('test.p'))
    
X_train, y_train = train['features'], train['labels']
X_test, y_test = test['features'], test['labels']


# Number of training examples
n_train = len(X_train) 

# Number of testing examples.
n_test = len(X_test) 

# What's the shape of an traffic sign image?
image_shape = X_train[0].shape 

# How many unique classes/labels there are in the dataset.
n_classes = len(set(y_train)) #set() returns unordered collection of unique elements

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)

"""## Visualizations"""

import matplotlib.pyplot as plt
# %matplotlib inline
import numpy as np
import random

#make a grid
fig, ax= plt.subplots(nrows=3, ncols=3)
fig.tight_layout()
flat = [axis for row in ax for axis in row]

#hide the messy ticks 
for axis in flat:
    axis.set_xticks([])
    axis.set_yticks([])
    axis.tick1On = axis.tick2On = False
    axis.label1On = axis.label2On = False

#include the image and the label
for i in range(3):
    for j in range(3):
        index = random.randint(0,len(X_train))
        img = X_train[index]
        class_id = y_train[index]
        ax[i][j].set_xlabel(class_id)
        ax[i][j].imshow(img)

plt.show()

def make_barchart(labels, string):
    samples_per_class = np.bincount(labels)
    x_tix = range(43)
    bins = np.arange(43) - 0.5
    fig = plt.figure(figsize=(25,4))
    plt.style.use('fivethirtyeight')
    ax1 = fig.add_subplot(1,1,1)
    plt.hist(y_train,bins)
    plt.title("Number of samples per different sign type" + string, loc='center')  
    plt.xlabel("Traffic Sign class/ids"); plt.ylabel("Samples ")
    plt.xticks(np.arange(min(x_tix), max(x_tix)+1, 1.0))
    plt.show();
    
make_barchart(y_train, "");

"""There is a csv file included in the directory which acts as a dictionary for sign id number and type of sign"""

import pandas as pd

names_ids = pd.read_csv('signnames.csv')
names_ids.head()

"""### Normalization
Normalization is an important part of pre-processing data so that everything is on a relative scale. For RGB images, pixel values range from 0 to 255 and are thus already normalized. However, we can apply further normalization to scale the values from 0 to 1 or -1 to 1. When possible, it's best to avoid large numbers because smaller numbers are both faster and more computationally stable.
"""

import cv2

def normalize(img):
    img_array = np.asarray(img)
    normalized = (img_array - img_array.min()) / (img_array.max() - img_array.min())
    return normalized

# X_train = histogram_equalization(X_train)
# X_test = histogram_equalization(X_test)

X_train = normalize(X_train)
X_test = normalize(X_test)

from sklearn.model_selection import train_test_split

#stratify param ensures same distribution of labels. important since there is a high variance in number of samples per label
#histograms of the new split sets will be the very, very similiar to the original set
X_train_split, X_validate_split, y_train_split, y_validate_split = train_test_split(X_train, y_train, 
                                                    test_size=0.2, 
                                                    random_state=666, 
                                                    stratify=y_train)

#training split
make_barchart(y_train_split, " in training split")

#validation split
make_barchart(y_validate_split, " in validation split")


import tensorflow as tf
from tensorflow.contrib.layers import flatten
from sklearn.utils import shuffle

X_train_split, y_train_split = shuffle(X_train_split, y_train_split)
X_validate_split, y_validate_split = shuffle(X_validate_split, y_validate_split)

EPOCHS = 150
BATCH_SIZE = 128
LEARNING_RATE = 0.001

x = tf.placeholder(tf.float32, (None, 32, 32, 3))
y = tf.placeholder(tf.int32, (None))
keep_prob = tf.placeholder(tf.float32)

#added after initial training in order to retrieve collection for testing new data in part 7
tf.add_to_collection('x', x)
tf.add_to_collection('y', y)
tf.add_to_collection('keep_prob', keep_prob)

#create one hot encoding of possible classes
one_hot_y = tf.one_hot(y, 43)

def LeNet(x):    
    # Hyperparameters
    mu = 0
    sigma = 0.1
    
    #Conv / ReLU / Max Pool
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 3, 32), mean = mu, stddev = sigma))
    conv1_b = tf.Variable(tf.zeros(32))
    conv1   = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b
    conv1 = tf.nn.relu(conv1)
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    #Conv / ReLU / Max Pool
    conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 32, 64), mean = mu, stddev = sigma))
    conv2_b = tf.Variable(tf.zeros(64))
    conv2   = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b
    conv2 = tf.nn.relu(conv2)
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    
    fc0 =  flatten(conv2)
       
    # Fully Connected / ReLu / Dropout
    fc1_W = tf.Variable(tf.truncated_normal(shape=(1600, 1024), mean = mu, stddev = sigma))
    fc1_b = tf.Variable(tf.zeros(1024))
    fc1 = tf.matmul(fc0, fc1_W) + fc1_b
    fc1 = tf.nn.relu(fc1)
    fc1_drop = tf.nn.dropout(fc1, keep_prob)

    # Fully Connected / ReLu / Dropout
    fc2_W  = tf.Variable(tf.truncated_normal(shape=(1024,512), mean = mu, stddev = sigma))
    fc2_b  = tf.Variable(tf.zeros(512))
    fc2 = tf.matmul(fc1_drop, fc2_W) + fc2_b
    fc2    = tf.nn.relu(fc2)
    fc2_drop = tf.nn.dropout(fc2, keep_prob)

    # Fully Connected 
    fc3_W  = tf.Variable(tf.truncated_normal(shape=(512, 43), mean = mu, stddev = sigma))
    fc3_b  = tf.Variable(tf.zeros(43))
    logits = tf.matmul(fc2_drop, fc3_W) + fc3_b
    weights = [conv1_W,conv2_W,fc1_W,fc2_W,fc3_W]
    
    #added after initial training in order to retrieve collection for testing new data in part 7
    tf.add_to_collection('logits', logits)
    
    return logits,weights

logits,weights = LeNet(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=one_hot_y)

# L2 Regularization 
regularizers = 0.0
for w in weights:
    regularizers += tf.nn.l2_loss(w)

# See notes below on l2 params
#http://docs.aws.amazon.com/machine-learning/latest/dg/training-parameters.html
L2_strength = 1e-6
loss_operation = tf.reduce_mean(cross_entropy) + L2_strength * regularizers
optimizer = tf.train.AdamOptimizer(learning_rate = LEARNING_RATE)
training_operation = optimizer.minimize(loss_operation)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#added after initial training in order to retrieve collection for testing new data in part 7
tf.add_to_collection('accuracy_operation', accuracy_operation)


def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    total_loss = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y, keep_prob:1.0})
        loss = sess.run(loss_operation, feed_dict={x: batch_x, y: batch_y, keep_prob:1.0})
        total_accuracy += (accuracy * len(batch_x))
        total_loss += (loss * len(batch_x))
    return total_accuracy / num_examples,total_loss / num_examples

config = tf.ConfigProto(allow_soft_placement = True)
config.gpu_options.per_process_gpu_memory_fraction=0.3

import os

save_dir = './model'

with tf.device('/gpu:0'):
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        num_examples = len(X_train_split)

        print("Training...")
        print()
        for i in range(EPOCHS):
            X_train_split, y_train_split = shuffle(X_train_split, y_train_split)
            for offset in range(0, num_examples, BATCH_SIZE):
                end = offset + BATCH_SIZE
                batch_x, batch_y = X_train_split[offset:end], y_train_split[offset:end]
                sess.run(training_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 0.5})
            training_accuracy,training_loss = evaluate(X_train_split, y_train_split)
            validation_accuracy,validation_loss = evaluate(X_validate_split, y_validate_split)

            if i%10 == 0:
                print("EPOCH {} ...".format(i))
                print("Training Accuracy = {:.4f} Validation Accuracy = {:.4f}".format(training_accuracy,validation_accuracy))
                print("Training Loss = {:.4f} Validation Loss = {:.4f}".format(training_loss,validation_loss))
                print()

        saver = tf.train.Saver()
        path = os.path.join(save_dir, 'lenet_traffic_classifier')
        saver.save(sess, path)
        print("Model saved")


#run model on testing sample
#tf.saver will not run on the GPU so have to soft_place the cpu device
with tf.device('/cpu:0'):
    with tf.Session(config=config) as sess:
        loader = tf.train.import_meta_graph('lenet_traffic_classifier.meta')
        loader.restore(sess, tf.train.latest_checkpoint(save_dir))

        X_test = normalize(X_test)

        test_accuracy,test_loss = evaluate(X_test, y_test)
        print("Test Accuracy = {:.3f}".format(test_accuracy))

import os
from skimage.transform import resize
import matplotlib.image as mpimg

filenames = os.listdir("test_photos/")

fig = plt.figure(figsize=(8,8))
fig.subplots_adjust(hspace=0.5)

for index,filename in enumerate(filenames):
    image = mpimg.imread('test_photos/'+filename)
    ax = fig.add_subplot(4,3,index+1)
    ax.set_xlabel(filename)
    image_resize = resize(image, (32, 32))
    plt.imshow(image_resize)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    for tic in ax.xaxis.get_major_ticks():
        tic.tick1On = tic.tick2On = False
        tic.label1On = tic.label2On = False
    for tic in ax.yaxis.get_major_ticks():
        tic.tick1On = tic.tick2On = False
        tic.label1On = tic.label2On = False


import numpy as np
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline

from skimage.transform import resize
import matplotlib.image as mpimg
import os

config = tf.ConfigProto(allow_soft_placement = True)
config.gpu_options.per_process_gpu_memory_fraction=0.3

filenames = os.listdir("test_photos/")

with tf.device('/cpu:0'):
    with tf.Session(config=config) as sess:
        loader = tf.train.import_meta_graph('model/lenet_traffic_classifier.meta')
        loader.restore(sess, tf.train.latest_checkpoint('model/'))
    
        accuracy_operation=tf.get_collection('accuracy_operation')[0] #returns list
        x=tf.get_collection('x')[0]
        y=tf.get_collection('y')[0]
        keep_prob=tf.get_collection('keep_prob')[0]
        logits=tf.get_collection('logits')[0]

        predictions =[]
  
        for index,filename in enumerate(filenames):
            image = mpimg.imread('test_photos/'+filename)
            test_image = resize(image, (32, 32))
            #pre-processing
            test_image = normalize(test_image)

            test_prediction = tf.nn.softmax(logits)
            classification = sess.run(test_prediction,feed_dict = {x: [test_image],keep_prob:1.0})
            test_class = sess.run(tf.argmax(classification,1))
            value,indices = sess.run(tf.nn.top_k(tf.constant(classification), k=3))

            predict_confidence=value.squeeze()
            indices = indices.squeeze()
            print(filename)
            fig = plt.figure(figsize=(1,1))
            plt.imshow(test_image)
            plt.axis('off')
            for j in range(0,3):
                print ( ' Class_id:{0:2d}  confidence:{1:.0%}'.format((indices[j]),(predict_confidence[j])))

            classes = indices.squeeze() 
            width = 0.75      
            fig = plt.figure(figsize=(6,1))
            ax = fig.add_subplot(1, 1, 1)

            rect = ax.bar(classes, predict_confidence*100, width,
                    color='green')

            # axes and labels
            ax.set_xlim(-width,len(classes)+width)
            ax.set_ylim(0,100)
            ax.set_ylabel('confidence in %')
            ax.set_title(' Top 3 scores')
            xTickMarks = ['class_id: '+str(classes[k]) for k in range(0,len(classes))]
            ax.set_xticks(classes+width)
            xtickNames = ax.set_xticklabels(xTickMarks)
            plt.setp(xtickNames, rotation=90, fontsize=8)
            plt.show()
            plt.close
