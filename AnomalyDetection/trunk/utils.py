# coding: utf8

import os
import csv
import traceback

import scipy.stats as stats
import matplotlib.pyplot as plt

import numpy as np
from pandas import read_csv

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

np.random.seed(27)



def clean_dir(path):
	for the_file in os.listdir(path):
	    file_path = os.path.join(path, the_file)
	    os.remove(file_path)

def random_split(dataX, dataY):
	'''
		Selects random samples from 
		normal and anomalous datasets

		Return :
			dataX : shuffled normal set
			dataY : shuffled anomalous set
	'''
	n_sampleX = int(dataX.shape[0]*0.5)
	n_sampleY = int(dataY.shape[0]*0.5)

	print('Number of normal samples :', n_sampleX)
	print('Number of anomalous samples :', n_sampleY)

	np.random.shuffle(dataX)
	np.random.shuffle(dataY)


	actual = np.hstack((np.zeros(dataX.shape[0]), np.ones(dataY.shape[0])))

	return np.vstack((dataX, dataY)), actual

def get_datasets(path_normal, path_anomalous):
	'''
		If the dataset is composed of two files (normal file and anomalous file)
		this function 
	'''

	# load the dataset
	normal_dataset = read_csv(path_normal, delimiter=',')
	anomalous_dataset = read_csv(path_anomalous, delimiter=',')

	labels = list(normal_dataset)
	print(labels)
	time = normal_dataset.values[:,0]

	print(time.shape)

	normal_dataset = normal_dataset.values[:, 1:]
	anomalous_dataset = anomalous_dataset.values[:, 1:]

	# split into train and test sets
	train_size = int(normal_dataset.shape[0] * 0.60)
	test_size = normal_dataset.shape[0] - int(train_size/2)
	val_size = normal_dataset.shape[0] - test_size
	train_data, test_data_normal = normal_dataset[0:train_size,:], normal_dataset[train_size:train_size+val_size,:]
	validation = normal_dataset[train_size+val_size:,:]

	# normalize the normal_dataset (with train scaler)
	scaler = MinMaxScaler(feature_range=(0, 1))

	train_data = scaler.fit_transform(train_data)
	test_data_normal = scaler.transform(test_data_normal)
	validation = scaler.transform(validation)
	test_data_anomalous = scaler.transform(anomalous_dataset)

	# Export normalized train_data to csv
	norm = np.vstack((time[:train_data.shape[0]], train_data.T)) 
	filename = 'norm_' + os.path.basename(path_normal)
	np.savetxt(filename, norm.T[:300,:], header=','.join(labels), delimiter=',')

	return train_data, validation, test_data_normal, test_data_anomalous
		

def fit_scaler(train_files):
	'''
		Fit a scaler on a list of files (usually train files)

		Agrument :
			train_files : list of train filepath

		Return :
			sc : fitted scaler
	'''
	sc = StandardScaler()
	
	# Fit scaler with every training files
	for filename in train_files:
		try:
			tmp = read_csv(filename, index_col=False).values[:, 1:]
			tmp = sc.partial_fit(tmp)
		except:
			print('[ ! ] Problem fitting scaler')
	return sc


def standardize(files, scaler, destination):
	'''
		Standardize all files in "files" and save the 
		result in "destination" using a fitted scaler

		Arguments:
			files : list of filepaths
			scaler : scaler used to standardize files
			destination : where standardized data is saved
	'''

	for in_file in files:

		out_file = destination + os.path.basename(in_file)
		
		try:
			# Get data and labels
			tmp = read_csv(in_file, index_col=False)
			labels = list(tmp)[1:]
			tmp = tmp.values[:, 1:]

			# Scale data
			tmp = scaler.transform(tmp)
			norm = np.vstack((np.arange(tmp.shape[0]), tmp.T)).T

			# Save 
			labels = ','.join(l for l in labels)
			np.savetxt(out_file, norm, delimiter=',', header=labels, comments='')
		
		except:
			print('[ ! ] Problem scaling :')
			print('[-->]', out_file)
			#traceback.print_exc()
			exit()


def standardize_dumb(files, destination):
	'''
		Standardize all files in "files" and save the 
		result in "destination"

		Arguments:
			files : list of filepaths
			destination : where standardized data is saved
	'''
	for in_file in files:

		out_file = destination + os.path.basename(in_file)
		
		try:
			# Get data and labels
			tmp = read_csv(in_file, index_col=False)
			labels = list(tmp)[1:]
			tmp = tmp.values[:, 1:]

			# Scale data
			tmp = StandardScaler().fit_transform(tmp)
			norm = np.vstack((np.arange(tmp.shape[0]), tmp.T)).T

			# Save 
			labels = ','.join(l for l in labels)
			np.savetxt(out_file, norm, delimiter=',', header=labels, comments='')
		
		except:
			print('[ ! ] Problem scaling :')
			print('[-->]', out_file)
			traceback.print_exc()
			exit()


def get_dataset_files_dumb(dataset_path):
	'''
		Each files are separately standardized

		Argument : 
			dataset_path : path to dataset

		Returns :
			train_files : list of train files paths
			test_files_n: list of normal test files paths
			test_files_a: list of anomalous test files paths

	'''
	print('[...] Retrieving datasets')

	# Get paths to data
	path_normal		= dataset_path + '/normal/not_normalized/'
	path_anomalous	= dataset_path + '/anomalous/not_normalized/'

	dest_normal		= dataset_path + '/normal/normalized/'
	dest_anomalous	= dataset_path + '/anomalous/normalized/'

	# Removes every normalized files if any
	clean_dir(dest_normal)
	clean_dir(dest_anomalous)

	# Compute normal set size
	nb_files = len(os.listdir(path_normal))
	train_nb = int(nb_files * 0.70)
	
	# Shuffle file list for cross validation WRONG
	l_file = os.listdir(path_normal)

	# Store sets file lists
	train_files = [path_normal+x for x in l_file[:train_nb]]
	test_files_n= [path_normal+x for x in l_file[train_nb:]]
	test_files_a= [path_anomalous+x for x in os.listdir(path_anomalous)]

	# Scale data
	print('[...] Scaling data')
	standardize_dumb(train_files, dest_normal)
	standardize_dumb(test_files_n, dest_normal)
	standardize_dumb(test_files_a, dest_anomalous)


	# Update files list
	train_files = [dest_normal+x for x in l_file[:train_nb]]
	test_files_n= [dest_normal+x for x in l_file[train_nb:]]
	test_files_a= [dest_anomalous+x for x in os.listdir(path_anomalous)]

	return train_files, test_files_n, test_files_a


def get_dataset_files(dataset_path):
	'''
		Return 
			train_files : train filepaths array
			test_files_n: normal test filepaths array
			test_files_a: anomalous test filespaths array
	'''
	print('[...] Retrieving datasets')

	# Get paths to data
	path_normal		= dataset_path + '/normal/not_normalized/'
	path_anomalous	= dataset_path + '/anomalous/not_normalized/'

	dest_normal		= dataset_path + '/normal/normalized/'
	dest_anomalous	= dataset_path + '/anomalous/normalized/'

	# Removes every normalized files if any
	clean_dir(dest_normal)
	clean_dir(dest_anomalous)

	# Compute normal set size
	nb_files = len(os.listdir(path_normal))
	train_nb = int(nb_files * 0.70)
	
	# Shuffle file list for cross validation
	l_file = os.listdir(path_normal)
	np.random.shuffle(np.asarray(l_file))

	# Store sets file lists
	train_files = [path_normal+x for x in l_file[:train_nb]]
	test_files_n= [path_normal+x for x in l_file[train_nb:]]
	test_files_a= [path_anomalous+x for x in os.listdir(path_anomalous)]

	# Scale data
	print('[...] Fitting scaler')
	scaler = fit_scaler(train_files)
	print('[...] Scaling data')
	standardize(train_files, scaler, dest_normal)
	standardize(test_files_n, scaler, dest_normal)
	standardize(test_files_a, scaler, dest_anomalous)

	# Update files list
	train_files = [dest_normal+x for x in l_file[:train_nb]]
	test_files_n= [dest_normal+x for x in l_file[train_nb:]]
	test_files_a= [dest_anomalous+x for x in os.listdir(path_anomalous)]

	return train_files, test_files_n, test_files_a


def drange(start, stop, increment):
	'''
		Define for loop with floats
	'''
	while start < stop:
		yield start
		start += increment


def save_roc(fp_rate, tp_rate, filename):
	'''
		Save roc curve as .png

		Parameters
			fp_rate = false positive rates array
			tp_rate = true positive rates array
	'''

	line_x = [0,1]

	plt.figure(figsize=(8, 8))

	plt.plot(fp_rate, tp_rate, color="blue", linewidth=1.0, linestyle="-")
	plt.plot(line_x, line_x, color="red", linewidth=1.0, linestyle="--")

	plt.title('ROC Curve')
	plt.xlabel('False positive rate')
	plt.ylabel('True positive rate')
	plt.axis([0, 1, 0, 1])

	plt.savefig(filename)


def save_losses(train_loss, valid_loss, anomal_loss, filename):
	'''
		Save plot of training loss over time as .png
	'''

	plt.figure(figsize=(12, 6))
	plt.plot(np.array(range(0, len(train_loss))) / float(len(train_loss) - 1) * (len(train_loss) - 1),
																								    np.log(train_loss),
																								    label="Train set loss")
	if valid_loss:
		plt.plot(np.log(valid_loss), label="Validation set loss")
	if anomal_loss:
		plt.plot(np.log(anomal_loss), label="Anomalous set loss")

	plt.title("Training errors over time (on a logarithmic scale)")
	plt.xlabel('Epochs')
	plt.ylabel('log(Loss)')
	plt.legend(loc='best')
	plt.savefig(filename)



def get_seq2seq_batch(data, seq_length, batch_size):
	'''
		Transpose data to shape (seq_length, batch_size, n_features)
		Used for recurrent models !
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

	batch = batch.transpose((1, 0, 2))

	return batch


def pad(matrix, reference):
	'''
		Pad bacth
	'''
	padded = np.ones(reference)
	padded[:matrix.shape[0], :matrix.shape[1]]
	return padded


def get_data_shape(file_list):
	'''
		Return data shape after transposing to (seq_length, batch_size, n_features)
	'''
	data = pd.read_csv(file_list[0]).values
	data = get_seq2seq_batch(data, seq_length, batch_size)
	return data.shape



class DatasetInfo(object):
	'''
		Collect information from a dataset folder
		Doesn't store any data
	'''

	def __init__(self, path_to_dataset):

		self.path = path_to_dataset


	def nb_features(self):

		path_normal = self.path + '/normal/normalized/'

		for filename in os.listdir(path_normal):
			file = path_normal + filename
			tmp = read_csv(file)
			return tmp.values.shape[1]-1

	def nb_normal_sample(self):
		path_normal = self.path + '/normal/normalized/'
		sum_sample = 0

		for filename in os.listdir(path_normal):
			file = path_normal + filename
			tmp = read_csv(file)

			sum_sample += tmp.values.shape[0]

		return sum_sample

	def nb_normal_files(self):
		path_normal = self.path + '/normal/normalized/'
		return len(os.listdir(path_normal))

	def nb_anomalous_sample(self):
		path_normal = self.path + '/anomalous/normalized/'
		sum_sample = 0

		for filename in os.listdir(path_anomalous):
			file = path_normal + filename
			tmp = read_csv(file)

			sum_sample += tmp.values.shape[0]

		return sum_sample

	def nb_anomalous_files(self):
		path_normal = self.path + '/anomalous/normalized/'
		return len(os.listdir(path_anomalous))

