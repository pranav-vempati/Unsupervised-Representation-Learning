
import tensorflow as tf   
import numpy as np 
import matplotlib.pyplot as plt 
from tensorflow.examples.tutorials.mnist import input_data

'''
TensorFlow implementation of a deep autoencoder
'''

num_displayed_images = 5

mnist = input_data.read_data_sets("MNIST_data", one_hot = True)

num_visible_units = 784 # MNIST data is comprised of a 28x28 grid of pixels

num_hidden1_units = 392

num_hidden2_units = 196

num_hidden3_units = 98

num_hidden4_units = num_hidden2_units

num_hidden5_units = num_hidden1_units

num_output_units = num_visible_units

visible_placeholder = tf.placeholder(tf.float32, shape = [None, num_visible_units]) # Define a placeholder for the input

w1 = tf.Variable(tf.truncated_normal([num_visible_units,num_hidden1_units]), dtype = tf.float32) # Layerwise initialization of the network's trainable weight matrices
w2 = tf.Variable(tf.truncated_normal([num_hidden1_units, num_hidden2_units]), dtype = tf.float32)
w3 = tf.Variable(tf.truncated_normal([num_hidden2_units, num_hidden3_units]), dtype = tf.float32)
w4 = tf.Variable(tf.truncated_normal([num_hidden3_units, num_hidden4_units]), dtype = tf.float32)
w5 = tf.Variable(tf.truncated_normal([num_hidden4_units, num_hidden5_units]), dtype = tf.float32)
w6 = tf.Variable(tf.truncated_normal([num_hidden5_units, num_output_units]), dtype = tf.float32)

b1 = tf.Variable(tf.zeros(num_hidden1_units)) # Layerwise intialization of the network's trainable bias vectors
b2 = tf.Variable(tf.zeros(num_hidden2_units))
b3 = tf.Variable(tf.zeros(num_hidden3_units))
b4 = tf.Variable(tf.zeros(num_hidden4_units))
b5 = tf.Variable(tf.zeros(num_hidden5_units))
b6 = tf.Variable(tf.zeros(num_output_units))

hidden_layer_1_activations = tf.nn.relu(tf.matmul(visible_placeholder, w1) + b1)
hidden_layer_2_activations = tf.nn.relu(tf.matmul(hidden_layer_1_activations, w2) + b2)
hidden_layer_3_activations = tf.nn.relu(tf.matmul(hidden_layer_2_activations, w3) + b3)
hidden_layer_4_activations = tf.nn.relu(tf.matmul(hidden_layer_3_activations, w4) + b4)
hidden_layer_5_activations = tf.nn.relu(tf.matmul(hidden_layer_4_activations, w5) + b5)
output = tf.nn.relu(tf.matmul(hidden_layer_5_activations, w6) + b6)

mse_loss = tf.reduce_mean(tf.square(output - visible_placeholder)) # MSE(mean-squared-error) reconstruction loss. Cross entropy loss tends to be asymmetric, which assigns erroneous extra penalty to some reconstructions

optimizer = tf.train.AdamOptimizer(0.01)

training = optimizer.minimize(mse_loss)

num_epochs = 20

batch_size = 100

num_examples = mnist.train.num_examples
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for epoch in range(num_epochs):
		for batch in range(int(num_examples//batch_size)):
			batch_examples, batch_targets = mnist.train.next_batch(batch_size)
			sess.run(training, feed_dict = {visible_placeholder:batch_examples})
		epochwise_loss = sess.run(mse_loss, feed_dict = {visible_placeholder:batch_examples})
		print("Epoch: " + str(epoch + 1) + " Loss: " + str(epochwise_loss)) 
	reconstructions = sess.run(output, feed_dict = {visible_placeholder:mnist.test.images[:num_displayed_images]})

	a, b = plt.subplots(2,10, figsize = (20,4))
	for index in range(num_displayed_images):
		b[0][index].imshow(np.reshape(mnist.test.images[index],(28,28)))
		b[1][index].imshow(np.reshape(reconstructions[index],(28,28)))












