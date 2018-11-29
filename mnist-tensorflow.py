from __future__ import division, print_function, absolute_import

import tensorflow as tf

# Make sure internet connection is ON for downloading MNIST dataset for first time.
# Import MNIST data from local or online repo
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Training Parameters
learning_rate = 0.001
num_steps = 1000
batch_size = 128
display_step = 50

# Network Parameters
num_input = 784 # MNIST data input (img shape: 28*28)
num_classes = 10 # MNIST total classes (0-9 digits)
dropout = 0.628 # Dropout, probability to keep units

# tf Graph input
X = tf.placeholder(tf.float32, [None, num_input])
Y = tf.placeholder(tf.float32, [None, num_classes])
keep_prob = tf.placeholder(tf.float32) # dropout (keep probability)

def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')

def conv_net(x, weights, biases, dropout):
    # MNIST data input is a 1-D vector of 784 features (28*28 pixels)
    # Reshape to match picture format [Height x Width x Channel]
    # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
    x = tf.reshape(x, shape=[-1, 28, 28, 1])

    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)

    # Convolution Layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=2)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Apply DropConnect for removing subsections of weights
    fc1 = tf.nn.dropout(fc1, keep_prob=dropout) * dropout
    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out

weights = {
    # 5x5 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([7*7*64, 1024])),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([1024, num_classes]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([num_classes]))
}

# Construct model
logits = conv_net(X, weights, biases, keep_prob)
prediction = tf.nn.softmax(logits)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)


# Evaluate model
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()


with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    for step in range(1, num_steps+1):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        # Run optimization op (backprop)
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y, keep_prob: dropout})
        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                 Y: batch_y,
                                                                 keep_prob: 1.0})
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))

    print("Optimization Finished!")

    # Calculate accuracy for 256 MNIST test images
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={X: mnist.test.images[:256],
                                      Y: mnist.test.labels[:256],
                                      keep_prob: 1.0}))
                                      
# Output Values

# Step 1, Minibatch Loss= 73474.3125, Training Accuracy= 0.055
# Step 50, Minibatch Loss= 3454.6426, Training Accuracy= 0.789
# Step 100, Minibatch Loss= 1958.1410, Training Accuracy= 0.875
# Step 150, Minibatch Loss= 1208.5380, Training Accuracy= 0.906
# Step 200, Minibatch Loss= 1469.4280, Training Accuracy= 0.891
# Step 250, Minibatch Loss= 654.0911, Training Accuracy= 0.953
# Step 300, Minibatch Loss= 639.9995, Training Accuracy= 0.969
# Step 350, Minibatch Loss= 729.3602, Training Accuracy= 0.930
# Step 400, Minibatch Loss= 432.9617, Training Accuracy= 0.969
# Step 450, Minibatch Loss= 1082.8853, Training Accuracy= 0.938
# Step 500, Minibatch Loss= 536.8420, Training Accuracy= 0.953
# Step 550, Minibatch Loss= 700.2826, Training Accuracy= 0.961
# Step 600, Minibatch Loss= 685.1751, Training Accuracy= 0.922
# Step 650, Minibatch Loss= 82.5561, Training Accuracy= 0.984
# Step 700, Minibatch Loss= 125.7426, Training Accuracy= 0.977
# Step 750, Minibatch Loss= 301.1901, Training Accuracy= 0.984
# Step 800, Minibatch Loss= 339.7703, Training Accuracy= 0.969
# Step 850, Minibatch Loss= 327.9889, Training Accuracy= 0.977
# Step 900, Minibatch Loss= 245.6316, Training Accuracy= 0.938
# Step 950, Minibatch Loss= 265.4570, Training Accuracy= 0.953
# Step 1000, Minibatch Loss= 170.3115, Training Accuracy= 0.969
# Optimization Finished!
# Testing Accuracy: 0.9765625               
