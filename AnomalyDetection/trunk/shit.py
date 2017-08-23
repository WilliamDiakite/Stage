# Autoencoder Class

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import traceback
import math


from sklearn.metrics import roc_curve, auc
from math import floor
from utils import drange, DatasetInfo
from pandas import read_csv



def variable_summaries(var):
  '''
  	Attach a lot of summaries to a Tensor (for TensorBoard visualization)
  '''
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)



class Autoencoder(object):

	def __init__(self, model_name, input_dim, hidden_layers, transfer_function=tf.nn.sigmoid, optimizer=tf.train.AdamOptimizer()):
		'''
			Init graph model

			Parameters :
				model_name	: name used for summaries and roc curve file
				input_dim	: nb of features in dataset
				hidden_layers: list of int that defines hidden layers architecture
				transfer_function : (activation function) 
				optimizer 	: gradient descent optimizer 
		'''

		#--- Model parameters
		self.model_name 	= model_name
		
		self.input_dim 		= input_dim
		self.hidden_layers	= hidden_layers
		self.n_output		= input_dim
		
		self.weights 		= dict()
		self.biases 		= dict()
		self.layers 		= dict() 
							
		self.transfer 		= transfer_function
		self.optimizer		= optimizer


		# Build the encoding layers
		self.input = tf.placeholder('float', [None, input_dim])
		next_layer_input = self.input

		encoding_matrices = []
		for dim in self.hidden_layers:
			input_dim = int(next_layer_input.get_shape()[1])

			# Initialize W using random values in interval [-1/sqrt(n) , 1/sqrt(n)]
			W = tf.Variable(tf.random_uniform([input_dim, dim], -1.0 / math.sqrt(input_dim), 1.0 / math.sqrt(input_dim)))

			# Initialize b to zero
			b = tf.Variable(tf.zeros([dim]))

			# We are going to use tied-weights so store the W matrix for later reference.
			encoding_matrices.append(W)

			output = tf.nn.tanh(tf.matmul(next_layer_input,W) + b)

			# the input into the next layer is the output of this layer
			next_layer_input = output

		# The fully encoded x value is now stored in the next_layer_input
		encoded_x = next_layer_input

		# build the reconstruction layers by reversing the reductions
		self.hidden_layers.reverse()
		encoding_matrices.reverse()


		for i, dim in enumerate(self.hidden_layers[1:] + [ int(self.input.get_shape()[1])]) :
			# we are using tied weights, so just lookup the encoding matrix for this step and transpose it
			W = tf.transpose(encoding_matrices[i])
			b = tf.Variable(tf.zeros([dim]))
			output = tf.nn.tanh(tf.matmul(next_layer_input,W) + b)
			next_layer_input = output

		# the fully encoded and reconstructed value of x is here:
		self.output = next_layer_input

		self.cost = tf.sqrt(tf.reduce_mean(tf.square(self.input-self.output)))
		self.optimizer = optimizer.minimize(self.cost)

		#--- Session init
		self.sess = tf.InteractiveSession()
		self.sess.run(tf.global_variables_initializer())
		self.saver = tf.train.Saver()


	#---------------------------------------------------#
	#				TRAINING & SCORING					#
	#---------------------------------------------------#
	
	
	def train_file(self, file, best_features, batch_size, start_file_prct, end_file_prct):
		
		# Read file
		tmp = read_csv(file)

		# Get start and stop marker
		start_file = int(tmp.shape[0]*start_file_prct)
		end_file = int(tmp.shape[0]*end_file_prct)

		if best_features is not None:
			tmp = tmp[best_features].values[start_file:end_file, 1:]
		else:
			tmp = tmp.values[start_file:end_file, 1:]

		batch_count = int(tmp.shape[0] / batch_size)
		current_pos = 0
		file_cost = 0.

		for i in range(batch_count+1):
			# Get next batch  		
			batch = tmp[i*batch_size:(i+1)*batch_size,]

			# Compute train step and loss
			self.sess.run([self.optimizer], feed_dict={self.input:batch}) 


	def train(self, train_files, validation_files=None, anomalous_files=None, best_features=None, batch_size=None, start_file_prct=0., end_file_prct=1., training_epochs=5):
		'''
			Train the model
			Parameters 
				train_files 	: list that links to every train files
				validation_files: list that links to every validation files (no training step)
				batch_size		: number of sample to feed before a training step
				start_file_prct : where the file should start (0 <= value <= 1)
				end_file_prct 	: where the file should stop (start_file <= value <= 1)
				training_epochs : number of times to feed entire dataset
				log_path 		: directory location to store summaries (tensorboard)
		'''

		train_losses = []
		valid_losses = []
		anomalous_losses = []

		# Start training
		for epoch in range(training_epochs):

			file_costs = []
			
			# Training using all files
			for file in train_files:
				self.train_file(file, best_features, batch_size, start_file_prct, end_file_prct)				
	
			# Display loss
			if epoch%10 == 0:
				
				train_loss = self.compute_set_loss(train_files, batch_size, best_features)
				train_losses.append(train_loss)

				if validation_files is not None:
					valid_loss = self.compute_set_loss(validation_files, batch_size, best_features)
					valid_losses.append(valid_loss)
				else:
					valid_loss = np.nan

				if anomalous_files is not None:
					anomalous_loss = self.compute_set_loss(anomalous_files, batch_size, best_features)
					anomalous_losses.append(anomalous_loss)
				else:
					anomalous_loss = np.nan

				print('\t[ + ] Step {}/{} \tTrain loss : {:.4f} \tValidation loss : {:.4f} \tAnomalous set loss : {:.4f}'.format(epoch+10, training_epochs, train_loss, valid_loss, anomalous_loss))

		# save model
		file = 'Saved_models/'+self.model_name+'.ckpt'
		save_path = self.saver.save(self.sess, file)
		print('[ + ] Model parameters have been saved')

		return train_losses, valid_losses, anomalous_losses


	def compute_set_loss(self, file_list, batch_size, best_features):
		'''
			Computes the average loss of a complete set (with multiple files)
			Parameters
				file_list : file list that links to every file of the dataset part
			Output 
				cost : average dataset cost
		'''
		file_costs = []

		# For every file in the train file list
		for file in file_list:

			# Read file
			data = read_csv(file)

			
			if best_features is not None:
				data = data[best_features].values[:, 1:]
			else:
				data = data.values[:, 1:]

			batch_count = int(data.shape[0] / batch_size)
	
			file_cost = 0.

			for i in range(batch_count):
				# Get next batc  		
				batch = data[i*batch_size:(i+1)*batch_size,]

				# Compute train step and loss
				batch_cost = self.sess.run([self.cost], feed_dict={self.input:batch}) 
				
				file_cost += batch_cost[0] / batch_count
			
			file_costs.append(file_cost)

		avg = np.mean(np.asarray(file_costs))
		return avg


	def predict(self, data, threshold):
		'''
			Computes the nb of positives and negatives in data for a given threshold
			Parameters 
				data 	  : samples to test
				threshold : detection threshold
			Output 
				n_positive: number of anomalous samples in data
				n_negative: number of normal samples in data
		'''
		n_positive = 0
		n_negative = 0

		for i in range(data.shape[0]):
			sample = np.reshape(data[i], (1, data.shape[1]))
			cost = self.sess.run(self.cost, feed_dict={self.input:sample})
			'''
			if i == 50:
				print(cost)
				print('Input',sample)
				print('Output\n',self.sess.run(self.output, feed_dict={self.input:sample}))
			'''
				

			if cost >= threshold:
				n_positive += 1
		
		n_negative = data.shape[0] - n_positive
		return n_positive, n_negative


	def get_roc(self, strt_thr, end_thr, step, files_test_normal, files_test_anomalous, best_features):

		print('[ + ] Computing ROC curve using :')
		print('\t[-->] Threshold from 0 to', end_thr)
		print('\t[-->] Step :', step)
		print('\t[-->] ROC curve has', int(end_thr/step),'points')

		point_count = 1 

		# Init empty arrays for points coordinates
		a_vp = []
		a_vn = []
		a_fp = []
		a_fn = []

		# For every threshold...
		for thr in drange(strt_thr, end_thr, step):
			tot_vn = 0
			tot_vp = 0
			tot_fn = 0
			tot_fp = 0
			n_sample_norm = 0
			n_sample_anom = 0

			# Compute predictions for normal test set
			for file in files_test_normal:

				# Read file
				data = read_csv(file, index_col=False)
				if best_features is not None:	
					data = data[best_features].values[:, 1:]
				else:
					data = data.values[:, 1:]

				# Predict file with current threshold
				fp, vn = self.predict(data, thr)
				
				# Update true/false positives count
				n_sample_norm += data.shape[0]
				tot_vn += vn
				tot_fp += fp


			# Compute predictions for anomalous test set
			for file in files_test_anomalous:

				# Read file
				data = read_csv(file, index_col=False)
				if best_features is not None:
					data = data[best_features].values[:, 1:]
				else:
					data = data.values[:, 1:]
				
				# Predict file with current threshold
				vp, fn = self.predict(data, thr)
				
				# Update 
				n_sample_anom += data.shape[0]
				tot_vp += vp
				tot_fn += fn

			vp_rate = tot_vp/n_sample_anom # Append for ROC curve
			vn_rate = tot_vn/n_sample_norm
			fp_rate = tot_fp/n_sample_norm # Append for ROC curve
			fn_rate = tot_fn/n_sample_norm

			print('\t[ + ] Point {}/{} \tTrue Positive Rate : {:.2f} \tFalse Positive Rate : {:.2f}'.format(point_count, 
																											int(end_thr/step),
																											vp_rate, fp_rate))
			point_count += 1
			
			a_vp.append(vp_rate)
			a_vn.append(vn_rate)
			a_fp.append(fp_rate)
			a_fn.append(fn_rate)

		return a_fp, a_vp
		

	def restore(self, path):
		# Restore variables from disk.
		self.saver.restore(self.sess, path)


	