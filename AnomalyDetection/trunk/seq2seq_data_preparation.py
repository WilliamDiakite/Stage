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
	nb_features = data.shape[1]-1

	# Sample to send to seq2seq
	batch = []

	for i in range(batch_size):
		seq = data[i*seq_length:(i+1)*seq_length, 1:]
		batch.append(seq)

		if seq.shape != (seq_length+1, nb_features):
			seq = pad(seq, reference=(seq_length+1, nb_features))
				
	batch = np.array(batch)
	#print('[ + ] Batch shape :', batch.shape)

	try :
		batch = batch.transpose((1, 0, 2))
	except:
		print('[ ! ] ERROR get_seq2seq_batch cannot transpose :', batch.shape)
		exit('[ ? ] Try lower batch_size and seq_length')
	#print('[ + ] Final batch shape :', batch.shape)

	return batch


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
	nb_features = data.shape[1]-1

	# Sample to send to seq2seq
	batch = []

	for i in range(batch_size):
		seq = data[i*seq_length:(i+1)*seq_length, 1:]
		batch.append(seq)

		if seq.shape != (seq_length+1, nb_features):
			seq = pad(seq, reference=(seq_length+1, nb_features))
				
	batch = np.array(batch)
	#print('[ + ] Batch shape :', batch.shape)

	try :
		batch = batch.transpose((0, 1, 2))
	except:
		print('[ ! ] ERROR get_seq2seq_batch cannot transpose :', batch.shape)
		exit('[ ? ] Try lower batch_size and seq_length')
	#print('[ + ] Final batch shape :', batch.shape)

	return batch



def pad(matrix, reference):
	padded = np.ones(reference)
	padded[:matrix.shape[0], :matrix.shape[1]]
	return padded