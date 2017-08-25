# coding: utf-8


import tensorflow as tf
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
plt.style.use('ggplot')

from seq2seq_data_preparation import get_seq2seq_batch
from utils import drange



class LSTM(object):
	def __init__(self, model_name, data_shape, hidden_dim, layers_stacked_count, learning_rate, lambda_l2_reg=0.003):


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
		self.layers_stacked_count = layers_stacked_count

		# Training hyperparameters
		self.learning_rate = learning_rate


		with tf.variable_scope('Seq2seq'):
		    # Encoder: inputs
		    self.enc_inp = [
		        tf.placeholder(tf.float32, shape=(
		            None, self.input_dim), name="inp_{}".format(t))
		        for t in range(self.seq_length)
		    ]

		    # Decoder: expected outputs
		    self.expected_sparse_output = [
		        tf.placeholder(tf.float32, shape=(None, self.output_dim),
		                       name="self.expected_sparse_output_".format(t))
		        for t in range(self.seq_length)
		    ]

		    # Give a "GO" token to the decoder.
		    # You might want to revise what is the appended value "+ self.enc_inp[:-1]".
		    dec_inp = [tf.zeros_like(
		        self.enc_inp[0], dtype=np.float32, name="GO")] + self.enc_inp[:-1]


		    # Create a `layers_stacked_count` of stacked RNNs (GRU cells here).
		    cells = []
		    for i in range(self.layers_stacked_count):
		        with tf.variable_scope('RNN_{}'.format(i)):
		            cells.append(tf.nn.rnn_cell.GRUCell(self.hidden_dim))
		            #cells.append(tf.nn.rnn_cell.BasicLSTMCell(self.hidden_dim))
		    cell = tf.nn.rnn_cell.MultiRNNCell(cells)


		    # For reshaping the input and output dimensions of the seq2seq RNN:
		    w_in  = tf.Variable(tf.random_normal([self.input_dim, self.hidden_dim]))
		    b_in  = tf.Variable(tf.random_normal([self.hidden_dim], mean=1.0))
		    w_out = tf.Variable(tf.random_normal([self.hidden_dim, self.output_dim]))
		    b_out = tf.Variable(tf.random_normal([self.output_dim]))

		    reshaped_inputs = [tf.nn.relu(tf.matmul(i, w_in) + b_in) for i in self.enc_inp]


		    # Here, the encoder and the decoder uses the same cell, HOWEVER,
		    # the weights aren't shared among the encoder and decoder, we have two
		    # sets of weights created under the hood according to that function's def.
		    dec_outputs, dec_memory = tf.nn.seq2seq.basic_rnn_seq2seq(
		        self.enc_inp,
		        dec_inp,
		        cell
		    )

		    output_scale_factor = tf.Variable(1.0, name="Output_ScaleFactor")
		    # Final outputs: with linear rescaling similar to batch norm,
		    # but without the "norm" part of batch normalization hehe.
		    self.reshaped_outputs = [output_scale_factor *
		                        (tf.matmul(i, w_out) + b_out) for i in dec_outputs]


		with tf.variable_scope('Loss'):
		    # L2 loss
		    output_loss = 0
		    for _y, _Y in zip(self.reshaped_outputs, self.expected_sparse_output):
		        output_loss += tf.reduce_mean(tf.nn.l2_loss(_y - _Y))

		    # L2 regularization (to avoid overfitting and to have a better
		    # generalization capacity)
		    reg_loss = 0
		    for tf_var in tf.trainable_variables():
		        if not ("Bias" in tf_var.name or "Output_" in tf_var.name):
		            reg_loss += tf.reduce_mean(tf.nn.l2_loss(tf_var))
		    
		    self.loss = output_loss + lambda_l2_reg * reg_loss
		    
		    

		with tf.variable_scope('Optimizer'):
			# Train operation
		    self.optimizer = tf.train.RMSPropOptimizer(learning_rate)
		    self.train_op = self.optimizer.minimize(self.loss)


		# Init session and variables
		self.sess = tf.InteractiveSession()
		self.sess.run(tf.global_variables_initializer())


	def train_batch(self, batch):
	    """
	    Training step that optimizes the weights
	    provided some batch_size X and Y examples from the dataset.
	    """
	    feed_dict = {self.enc_inp[t]: batch[t] for t in range(len(self.enc_inp))}
	    feed_dict.update({self.expected_sparse_output[t]: batch[t] for t in range(len(self.expected_sparse_output))})
	    _, loss_t = self.sess.run([self.train_op, self.loss], feed_dict)

	    return loss_t


	def test_batch(self, batch):
		"""
		Test step, does NOT optimizes. Weights are frozen by not
		doing sess.run on the train_op.
		"""
		# If there's only one sample in batch, need reshape
		if batch.ndim == 2:
			batch = np.reshape(batch, (batch.shape[0], 1 , batch.shape[1]))

		if batch.ndim == 1:
			batch = np.reshape(batch, (self.seq_length, 1, batch.shape[0]))

		
		feed_dict = {self.enc_inp[t]: batch[t] for t in range(len(self.enc_inp))}
		feed_dict.update({self.expected_sparse_output[t]: batch[t] for t in range(len(self.expected_sparse_output))})
		loss_t = self.sess.run([self.loss], feed_dict)
		

		return loss_t[0] / batch.shape[1]


	def predict_batch(self, batch):
		'''
			Computes prediction of a batch
		'''
		feed_dict = {self.enc_inp[t]: batch[t] for t in range(self.seq_length)}
		outputs = self.sess.run([self.reshaped_outputs], feed_dict)[0]
		return outputs


	def train(self, train_file_list, validation_file_list=None, anomalous_file_list=None, best_features=None, epoch=200):

		print('[ + ] Starting training !')

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

				#print('One sample time step : ', data[0,0,:])
				#print('Prediction', self.predict_batch(data)[0,0,:])

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

		# For every sequence of the batch
		for seq in range(data.shape[1]):
			seq_loss = self.test_batch(data[:,seq,:])

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
		data_matrix = data.values[:,:1]

	return data_matrix