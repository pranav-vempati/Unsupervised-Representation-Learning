import tensorflow as tf   
import numpy as np 
import matplotlib.pyplot as plt 
from tensorflow.examples.tutorials.mnist import input_data

num_displayed_images = 5

mnist = input_data.read_data_sets("MNIST_data", one_hot = True)

num_visible_units = 784 # MNIST data is comprised of a 28x28 grid of pixels

num_hidden1_units = 392

num_hidden2_units = 196

num_hidden3_units = 98

num_hidden4_units = num_hidden2_units

num_hidden5_units = num_hidden1_units

num_output_units = num_visible_units

visible_placeholder = tf.placeholder(tf.float32, shape = [None, num_visible_units]) # Define a placeholder for the MNIST images 

def weight_matrix(input_dim, output_dim):
	return tf.Variable(tf.truncated_normal([input_dim, output_dim]), dtype = tf.float32)

def bias_vector(output_dim):
	return tf.Variable(tf.zeros([output_dim]))


def log_tensor(tensor):
	with tf.name_scope('consolidated_summaries'):
		mean = tf.reduce_mean(tensor)
		tf.summary.scalar('mean', mean)
	with tf.name_scope('standard_deviation'):
		standard_deviation = tf.sqrt(tf.reduce_mean(tf.square(tensor-mean)))
	tf.summary.scalar('standard_deviation', standard_deviation)
	tf.summary.histogram('histogram', tensor)

def network_layer(incoming_tensor, input_dim, output_dim, layer_name):
	with tf.name_scope(layer_name):
		with tf.name_scope("Weights"):
			weights = weight_matrix(input_dim, output_dim)
			log_tensor(weights)
		with tf.name_scope("Biases"):
			biases = bias_vector(output_dim)
			log_tensor(biases)
		activations = tf.nn.relu(tf.matmul(incoming_tensor, weights) + biases)
		tf.summary.histogram('activations', activations)
		return activations

def corrupt_inputs(pristine_inputs, additive_noise): # Inject a tensor encoding a Gaussian distribution of zero mean and unit variance to corrupt the inputs
	return pristine_inputs + additive_noise

gaussian_noise = tf.random_normal([num_visible_units], mean = 0.0, stddev = 1.0, dtype = tf.float32)

corrupted_inputs = corrupt_inputs(visible_placeholder, gaussian_noise)

first_layer = network_layer(corrupted_inputs, num_visible_units, num_hidden1_units, "First_Layer")
second_layer = network_layer(first_layer, num_hidden1_units, num_hidden2_units, "Second_Layer")
third_layer = network_layer(second_layer, num_hidden2_units, num_hidden3_units, "Third_Layer")
fourth_layer = network_layer(third_layer, num_hidden3_units, num_hidden4_units, "Fourth_Layer")
fifth_layer = network_layer(fourth_layer, num_hidden4_units, num_hidden5_units, "Fifth_Layer")
outputs = network_layer(fifth_layer, num_hidden5_units, num_output_units, "Output_Layer")

with tf.name_scope("MSE_Loss"):
	mse_loss = tf.reduce_mean(tf.square(outputs-visible_placeholder)) # Element wise reconstruction loss between the values emitted by the autoencoder and the original inputs
tf.summary.scalar('MSE_Loss', mse_loss)

with tf.name_scope("training"):
	trainer = tf.train.AdamOptimizer(0.01).minimize(mse_loss)

merged = tf.summary.merge_all()

num_epochs = 12

batch_size = 100

num_examples = mnist.train.num_examples

num_batches = int(num_examples//batch_size)

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	event_writer = tf.summary.FileWriter("./TensorBoardLogs", graph = sess.graph)
	for epoch in range(num_epochs):
		aggregate_batch_loss = 0
		for batch in range(num_batches):
			batch_examples, batch_targets = mnist.train.next_batch(batch_size)
			batch_summary, _, batch_loss = sess.run([merged,trainer,mse_loss], feed_dict = {visible_placeholder:batch_examples})
			aggregate_batch_loss += batch_loss
			event_writer.add_summary(batch_summary, num_batches*epoch + batch)
		epochwise_loss = (aggregate_batch_loss/num_batches)
		print("Epoch: " + str(epoch + 1) + " Loss: " + str(epochwise_loss)) # Logs loss to stdout at every epoch
	training_summary,reconstructions = sess.run([merged,outputs], feed_dict = {visible_placeholder:mnist.test.images[:num_displayed_images]})


	a, b = plt.subplots(2,10, figsize = (20,4))
	for index in range(num_displayed_images):
		b[0][index].imshow(np.reshape(mnist.test.images[index],(28,28)))
		b[1][index].imshow(np.reshape(reconstructions[index],(28,28)))







