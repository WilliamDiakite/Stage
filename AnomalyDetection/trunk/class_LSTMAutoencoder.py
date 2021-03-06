# coding: utf-8


import tensorflow as tf
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
plt.style.use('ggplot')

from seq2seq_data_preparation import get_seq2seq_batch
from utils import drange



class LSTMAutoencoder(object):
	def __init__(self, model_name, data_shape, hidden_dim, learning_rate,):


		# Backward compatibility for TensorFlow's version 0.12:
		try:
		    tf.nn.seq2seq = tf.contrib.legacy_seq2seq
		    tf.nn.rnn_cell = tf.contrib.rnn
		    tf.nn.rnn_cell.GRUCell = tf.contrib.rnn.GRUCell
		    print("TensorFlow's version : 1.0 (or more)")
		except:
		    print("TensorFlow's version : 0.12")


		self.model_name = model_name
		
		self.batch_size = data_shape[1]
		self.seq_length = data_shape[0]

		# Architecture parameters
		self.input_dim = data_shape[2]
		self.output_dim= data_shape[2]
		self.hidden_dim= hidden_dim

		self.learning_rate = learning_rate
		

		initializer = tf.random_uniform_initializer(-1, 1)

		# the sequences, has n steps of maximum size
		self.seq_input = tf.placeholder(tf.float32, [self.seq_length, self.batch_size, self.input_dim])
		# what timesteps we want to stop at, notice it's different for each batch hence dimension of [batch]
		# early_stop = tf.placeholder(tf.int32, [batch_size])

		# inputs for rnn needs to be a list, each item/frame being a timestep.
		# we need to split our input into each timestep, and reshape it because split keeps dims by default
		encoder_inputs = [tf.reshape(self.seq_input, [-1, self.input_dim])]
		# if encoder input is "X, Y, Z", then decoder input is "0, X, Y, Z". Therefore, the decoder size
		# and target size equal encoder size plus 1. For simplicity, here I droped the last one.
		decoder_inputs = ([tf.zeros_like(encoder_inputs[0], name="GO")] + encoder_inputs[:-1])
		targets = encoder_inputs
		weights = [tf.ones_like(targets_t, dtype=tf.float32) for targets_t in targets]

		# basic LSTM seq2seq model
		cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_dim)
		_, enc_state = tf.contrib.rnn.static_rnn(cell, encoder_inputs, dtype=tf.float32)
		cell = tf.nn.rnn_cell.OutputProjectionWrapper(cell, self.input_dim)
		dec_outputs, dec_state = tf.nn.seq2seq.rnn_decoder(decoder_inputs, enc_state, cell)

		# flatten the prediction and target to compute squared error loss
		y_true = [tf.reshape(encoder_input, [-1]) for encoder_input in encoder_inputs]
		y_pred = [tf.reshape(dec_output, [-1]) for dec_output in dec_outputs]

		# Define loss and optimizer, minimize the squared error
		self.loss = 0
		for i in range(len(y_true)):
		    self.loss += tf.reduce_sum(tf.square(y_pred[i] - y_true[i]))
		self.train_op = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)

		# Initializing the variables
		self.sess = tf.Session()
		self.sess.run(tf.global_variables_initializer())


	def train_batch(self, batch):
	    """
	    Training step that optimizes the weights
	    """
	    _, loss_t = self.sess.run([self.train_op, self.loss], feed_dict={self.seq_input:batch})
	    return loss_t


	def test_batch(self, batch):
		"""
		Test step, does NOT optimizes. Weights are frozen by not
		doing sess.run on the train_op.
		"""
		# If there's only one sample in batch, need reshape
		if batch.ndim == 2:
			#batch = np.reshape(batch, (batch.shape[0], 1 , batch.shape[1]))
			pass

		if batch.ndim == 1:
			#batch = np.reshape(batch, (self.seq_length, self, batch.shape[0]))
			pass

		loss_t = self.sess.run([self.loss], feed_dict={self.seq_input:batch})
		return loss_t[0] 

	'''
	def predict_batch(self, batch):
		
		#Computes prediction of a batch
		feed_dict = {self.enc_inp[t]: batch[t] for t in range(self.seq_length)}
		outputs = self.sess.run([self.reshaped_outputs], feed_dict)[0]
		return outputs
	'''


	def train(self, train_file_list, validation_file_list=None, anomalous_file_list=None, best_features=None, epoch=200):

		print('[ + ] Starting training !')

		skipped_files = 0

		avg_train_losses = []
		avg_valid_losses = []
		avg_anomalous_losses = []

		for e in range(epoch):

			losses = []

			for file in train_file_list:

				# Load data
				data = pd.read_csv(file)
				data = keep_best_features(data, best_features)

				# Prepare data for seq2seq processing
				data = get_seq2seq_batch(data, self.seq_length, self.batch_size)

				# Train step
				loss = self.train_batch(data)
				losses.append(loss)
				
				
			# Compute losses to display
			if e%10 == 0:

				# Computes average loss on train files
				avg_loss = np.average(np.asarray(losses))
				avg_train_losses.append(avg_loss)
				
				# Compute validation loss
				if validation_file_list is not None:
					avg_val_loss = self.compute_set_loss(validation_file_list, best_features)
					avg_valid_losses.append(avg_val_loss)
				else:
					avg_val_loss = np.nan
				
				# Compute anomlous set loss
				if anomalous_file_list is not None:
					avg_ano_loss = self.compute_set_loss(anomalous_file_list, best_features)
					avg_anomalous_losses.append(avg_ano_loss)
				else:
					avg_ano_loss = np.nan

				# Display losses
				print('\t[ + ] Step {}/{} \tTrain loss : {:.4f} \tValidation loss : {:.4f} \tAnomalous set loss : {:.4f}'.format(e+10, epoch, avg_loss, avg_val_loss, avg_ano_loss))

		print('[ + ] Skipped files :', skipped_files)
		return avg_train_losses, avg_valid_losses, avg_anomalous_losses


	def compute_set_loss(self, file_list, best_features):
		losses = []
					
		for file in file_list:
			# Load data
			data = pd.read_csv(file)
			data = keep_best_features(data, best_features)

			# Prepare data for seq2seq processing
			data = get_seq2seq_batch(data, self.seq_length, self.batch_size)

			# Test data 
			loss = self.test_batch(data)
			losses.append(loss)

		# Computes average loss on validation files
		avg_loss = np.average(np.asarray(losses))
		return avg_loss
		


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
				
		seq_loss = self.test_batch(data)

		if seq_loss > threshold:
			n_positive += 1
		else:
			n_negative += 1

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

				# Load data
				data = pd.read_csv(file)
				data = keep_best_features(data, best_features)

				# Prepare data for seq2seq processing
				data = get_seq2seq_batch(data, self.seq_length, self.batch_size)

				# Predict file with current threshold
				fp, vn = self.predict(data, thr)
				
				# Update true/false positives count
				n_sample_norm += data.shape[1]
				tot_vn += vn
				tot_fp += fp


			# Compute predictions for anomalous test set
			for file in files_test_anomalous:

				# Load data
				data = pd.read_csv(file)
				data = keep_best_features(data, best_features)

				# Prepare data for seq2seq processing
				data = get_seq2seq_batch(data, self.seq_length, self.batch_size)

				# Predict file with current threshold
				vp, fn = self.predict(data, thr)
				
				# Update 
				n_sample_anom += data.shape[1]
				tot_vp += vp
				tot_fn += fn

			vp_rate = tot_vp/n_sample_anom *10 # Append for ROC curve
			vn_rate = tot_vn/n_sample_norm *10
			fp_rate = tot_fp/n_sample_norm *10 # Append for ROC curve
			fn_rate = tot_fn/n_sample_norm *10

			print('\t[ + ] Point {}/{} \tTrue Positive Rate : {:.5f} \tFalse Positive Rate : {:.5f}'.format(point_count, 
																											int(end_thr/step),
																											vp_rate, fp_rate))
			point_count += 1
			
			a_vp.append(vp_rate)
			a_vn.append(vn_rate)
			a_fp.append(fp_rate)
			a_fn.append(fn_rate)

		return a_fp, a_vp




def keep_best_features(data, best_features):
	'''
		Drop not wanted features if 
		feature selection has been set
	'''
	if best_features is not None:
		try :
			data_matrix = data[best_features].values
		except:
			print('[ ! ] ERROR in keep_best_features : keeping best features')
			print('[ ! ] best features :', best_features)
			print('[ ! ] data labels :', list(data))
			#traceback.print_exc()
			exit()
	else:
		# Remove index column
		data_matrix = data.values

	return data_matrix