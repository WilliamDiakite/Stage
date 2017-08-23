# coding: utf8

# Autoencoder Class

import os
import math
import numpy as np

import tensorflow as tf

import matplotlib.pyplot as plt
import traceback

from sklearn.metrics import roc_curve, auc
from math import floor
from utils import drange, DatasetInfo
from pandas import read_csv



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
		self.shape			= [self.input_dim] + hidden_layers + [self.input_dim]
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
	

	def train_file(self, data, best_features, batch_size):
		'''
			Train a single file 
		'''
		cost = 0

		# Keep best features only
		if best_features is not None:
			data = data[best_features].values
		else:
			data = data.values

		batch_count = int(data.shape[0]/batch_size)

		# Train model with every batch from current file
		for i in range(batch_count):		
			batch = data[i*batch_size:(i+1)*batch_size,:]
			
			_, batch_cost = self.sess.run([self.optimizer, self.cost], feed_dict={self.input:batch})
			
			cost += batch_cost/batch_count
			
		return cost

			
	def train(self, train_files, validation_files=None, anomalous_files=None, best_features=None, batch_size=50, start_file_prct=0, end_file_prct=1, training_epochs=5, log_path='./Summaries/'):
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
		# Init loss curves
		train_losses = []
		valid_losses = []
		anomalous_losses = []

		# Start training
		for epoch in range(training_epochs):

			file_costs = []

			# For every file in the train file list
			for file in train_files:

				# Read file
				data = read_csv(file, index_col=False)

				# Get start and stop marker
				start_file = int(data.shape[0]*start_file_prct)
				end_file = int(data.shape[0]*end_file_prct)
				data = data[start_file:end_file]

				# Train file et compute loss on file
				self.train_file(data, best_features, batch_size)
				

			if epoch%10 == 0:
				# Compute train average loss 
				train_loss = self.compute_set_loss(train_files, best_features)
				train_losses.append(train_loss)

				# Compute validation average loss if file list provided
				if validation_files is not None:
					valid_loss = self.compute_set_loss(validation_files, best_features)
					valid_losses.append(valid_loss)
				else:
					valid_loss = np.nan

				# Compute anomalous average loss if file list provided
				if anomalous_files is not None:
					anomalous_loss = self.compute_set_loss(anomalous_files, best_features)
					anomalous_losses.append(anomalous_loss)
				else:
					anomalous_loss = np.nan

				print('\t[ + ] Step {}/{} \tTrain loss : {:.4f} \tValidation loss : {:.4f} \tAnomalous set loss : {:.4f}'.format(epoch+10, training_epochs, train_loss, valid_loss, anomalous_loss))

		# save model
		file = 'Saved_models/'+self.model_name+'.ckpt'
		save_path = self.saver.save(self.sess, file)
		print('[ + ] Model parameters have been saved')

		return train_losses, valid_losses, anomalous_losses


	def compute_set_loss(self, file_list, best_features):
		'''
			Computes the average loss of a complete set (with multiple files)
			Parameters
				file_list : file list that links to every file of the dataset part
			Output 
				cost : average dataset cost
		'''
		cost = 0.

		for file in file_list:

			# Read file 
			data = read_csv(file, index_col=False)
			
			if best_features is not None:
				data = data[best_features].values
			else:
				data = data.values

			# Compute file cost
			file_cost = self.sess.run(self.cost, feed_dict={self.input:data}) 
			cost += (file_cost / len(file_list))				
		return cost


	def predict(self, data, threshold, batch_size):
		'''
			Computes the nb of positives and negatives in data for a given threshold
			Parameters 
				data 	  : samples to test
				threshold : detection threshold
				batch_size: if > 1, the threshold is compared the average batch loss 
			Output 
				n_positive: number of anomalous samples in data
				n_negative: number of normal samples in data
		'''

		n_positive = 0
		n_negative = 0

		#--- Sing sample prediction
		if batch_size == 1:
			for i in range(data.shape[0]):
				sample = np.reshape(data[i], (1, data.shape[1]))
				cost = self.sess.run(self.cost, feed_dict={self.input:sample})

				if cost >= threshold:
					n_positive += 1

		#--- Batch prediction
		elif batch_size > 1:
			# Compute number of batches
			batch_count = int(data.shape[0] / batch_size)+1

			for i in range(batch_count):			
				# Get batch
				batch = data[i*batch_size:(i+1)*batch_size,:]
				batch_cost = self.sess.run(self.cost, feed_dict={self.input:batch}) 
				if batch_cost > threshold:
					n_positive += batch_size
				

		#--- Batch size error
		else:
			exit('[ ! ] ERROR in predict() : Score batch size must be > or = to 1')

		n_negative = data.shape[0] - n_positive
		return n_positive, n_negative


	def get_roc(self, strt_thr, end_thr, step, files_test_normal, files_test_anomalous, batch_size, best_features):

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
					try :
						data = data[best_features].values
					except:
						print('[ ! ] ERROR in plot_roc : keeping best features:')
						print('[ ! ] best features :', best_features)
						traceback.print_exc()
						exit()
				else:
					# Removes index column
					data = data.values[:, 1:]

				# Predict file with current threshold
				fp, vn = self.predict(data, thr, batch_size)
				
				# Update true/false positives count
				n_sample_norm += data.shape[0]
				tot_vn += vn
				tot_fp += fp


			# Compute predictions for anomalous test set
			for file in files_test_anomalous:

				# Read file
				data = read_csv(file, index_col=False)
				if best_features is not None:
					try :
						data = data[best_features].values
					except Exception:
						print('[ ! ] ERROR in plot_roc : keeping best features')
						print('[ ! ] best features :', best_features)
						traceback.print_exc()
						exit()
				else:
					# Removes index column
					data = data.values[:, 1:]
				
				# Predict file with current threshold
				vp, fn = self.predict(data, thr, batch_size)
				
				# Update 
				n_sample_anom += data.shape[0]
				tot_vp += vp
				tot_fn += fn

			vp_rate = tot_vp/n_sample_anom # Append for ROC curve
			vn_rate = tot_vn/n_sample_norm
			fp_rate = tot_fp/n_sample_norm # Append for ROC curve
			fn_rate = tot_fn/n_sample_norm

			print('\t[ + ] Point {}/{} \tTrue Positive Rate : {:.5f} \tFalse Positive Rate : {:.5f}'.format(point_count, 
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






	#---------------------------------------------------#
	#					FOR DEBUG 						#
	#---------------------------------------------------#


	# Use if train set is one big csv file
	def train_debug(self, X, batch_size, training_epochs):

		total_batch = int(X.shape[0] / batch_size)
		
		for epoch in range(training_epochs):
			total_loss  = 0.
			total_loss_cost_bh = 0.
			current_pos = 0

			for b in range(total_batch+1):
				
				batch = tmp[b*batch_size:(b+1)*batch_size,:]

				_, loss = self.sess.run([self.optimizer, self.cost], feed_dict={self.input:batch})

				'''
				#--- Debug
				origin, recons = self.sess.run([self.input, self.output], feed_dict={self.input:batch})			
				cost_bh = self.sess.run(self.cost_bh, feed_dict={self.A:origin, self.B:recons})
				total_loss_cost_bh += (cost_bh / total_batch)
				#----------
				'''

				total_loss += (loss / total_batch)
				#print('batch loss :', loss)

			print('EPOCH ', epoch+1, '/', training_epochs)
			print('Loss :', total_loss)
			print('Cost by hand :', total_loss_cost_bh)

	# Use if test set is one big csv file
	def plot_roc_simple(self, end_thr, step, test_normal, test_anomalous):
		a_vp = []
		a_vn = []
		a_fp = []
		a_fn = []

		n_sample_norm = test_normal.shape[0]
		n_sample_anom = test_anomalous.shape[0]

		for i in drange(0, end_thr, step):
			
			# Predict file with current threshold
			fp, vn = self.predict(test_normal, i)
			
			# Predict file with current threshold
			vp, fn = self.predict(test_anomalous, i)
			
			# Compute rates
			vp_rate = vp/n_sample_anom #
			vn_rate = vn/n_sample_norm
			fp_rate = fp/n_sample_norm #
			fn_rate = fn/n_sample_anom

			print('-- threshold =', i)
			print('fp_rate :', fp_rate)
			print('vp_rate :', vp_rate)

			# Append points for ROC curve
			a_vp.append(vp_rate)
			a_vn.append(vn_rate)
			a_fp.append(fp_rate)
			a_fn.append(fn_rate)

		# Saving file
		filename = './roc/' + self.model_name + '.png'
		plot(a_fp, a_vp, filename)

		return a_fp, a_vp