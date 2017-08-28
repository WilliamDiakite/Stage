# coding: utf8

import numpy as np
import pandas as pd


def get_seq2seq_batch(data, seq_length, batch_size):
	'''
		Turns data with shape (num_samples, num_features)
		into data with shape (seq_legnth, batch_size, num_features)

		Arguments
			data : original data
			seq_length : length of the output sequences
			batch_size : number of sequence to store 

		Returns 
			batch : new data as seq2seq batch

		(warning) non complete sequences are padded with zeros
	'''
	
	# Read file
	nb_features = data.shape[1]

	# Sample to send to seq2seq
	batch = []

	for i in range(batch_size):
		seq = data[i*seq_length:(i+1)*seq_length,]
		
		if seq.shape != (seq_length, nb_features):
			seq = pad(seq, reference=(seq_length, nb_features))
			batch.append(seq)
		else:
			batch.append(seq)
	
	batch = np.array(batch)
	batch = batch.transpose((1, 0, 2))
	
	return batch


def pad(matrix, reference):
	padded = np.ones(reference)
	padded[:matrix.shape[0], :matrix.shape[1]]
	return padded

