# coding: utf-8
# test seq2seq class

import os
import sys
import time 

import numpy as np
import pandas as pd

import utils
import feature_selection

import matplotlib.pyplot as plt
plt.style.use('ggplot')

from beautifultable import BeautifulTable
from class_seq2seq import seq2seq 
from seq2seq_data_preparation import get_seq2seq_batch



# Path to "normal" dataset
dataset_path 	= './../../Ressources/Generated_files/Datasets/fast'

#------ Hyperparameters ------#

# Architecture
seq_length = hidden_dim = 20
layers_stacked_count = 5

# Learning parameters
learning_rate = 0.0007
lr_decay = 0.92
momentum = 0.9

epoch = 20
batch_size = 10
#-----------------------------#


#--- Detection parameters  ---#

start_thr = 0
end_thr = 700
step_thr = 140
#-----------------------------#


#--- Feature selection ---#
k_best = 5
min_feature_importance = 0.5
#-------------------------#



def get_data_shape(file_list):
	try:
		data = pd.read_csv(file_list[0]).values
	except:
		exit('[ ! ] ERROR get_data_shape(..) : reading data failed')

	data = get_seq2seq_batch(data, seq_length, batch_size)
	return data.shape


def write_summary(train_loss, valid_loss, anomal_loss, 
				start_thr, end_thr, false_positives, true_positives,
				training_time, best_features, io_dim):
	'''
		Saves losses curves, roc curves and model parameters 
	'''
	# Init directory
	summary_dir = './Summaries/'
	model_dir = summary_dir + model_name + '/'
	while os.path.exists(model_dir):
		print('[ ! ] Model summary already exists.')
		tmp_name = input('[ ? ] Choose new name for model directory: ')
		model_dir = summary_dir + tmp_name + '/'
	os.makedirs(model_dir)

	# Save loss curves
	loss_name = model_dir + 'losses.png'
	utils.save_losses(train_loss, valid_loss, anomal_loss, loss_name)

	# Save ROC curve
	roc_name = model_dir + 'roc.png'
	utils.save_roc(false_positives, true_positives, roc_name)

	# Save parameters file
	parameters_file = model_dir + 'parameters.txt'
	parameters = open(parameters_file, 'w')

	parameters.write('Document creation : {}\n'.format(time.strftime("%c")) )
	parameters.write('Model name : {}\n'.format(model_name))
	parameters.write('Model type : seq2seq\n')
	parameters.write('Dataset : {}\n'.format(dataset_path))

	parameters.write('\nSequence length : {}\n'.format(seq_length))	
	parameters.write('Sequence dimension : {}'.format(io_dim))
	parameters.write('Stacked layers count : {}\n'.format(layers_stacked_count))
	parameters.write('Batch size : {}\n'.format(batch_size))
	parameters.write('Epochs : {}\n'.format(epoch))

	parameters.write('\nOptimizer : RMSprop\n')
	parameters.write('Learning rate decay : {}\n'.format(lr_decay))
	parameters.write('Learning rate : {}\n'.format(learning_rate))
	parameters.write('Momentum : {}\n'.format(momentum))

	parameters.write('\nTraining time : {}\n'.format(time.strftime('%H:%M:%S', time.localtime(training_time))))

	parameters.write('\nROC settings :\n')
	parameters.write('\tThreshold start : {}\n'.format(start_thr))
	parameters.write('\tThreshold end : {}\n'.format(end_thr))
	parameters.write('\tStep : {}\n'.format(step_thr))

	if best_features is not None:
		parameters.write('\nFeature selection : \n')
		# Change std to print directly in file
		old_std = sys.stdout
		sys.stdout = parameters
		# Draw table
		table = BeautifulTable()
		table.column_headers = ['Selected features']
		for feature in best_features:
			table.append_row([feature])
		print(table)
		# restore original stdout
		sys.stdout = old_std

	parameters.close()
	print('[ + ] Summary saved !')



if __name__ == '__main__':

	from argparse import ArgumentParser
	parser = ArgumentParser()
	parser.add_argument('-feature_select', action='store_true', help='Apply feature selection, only keep best features during training and testing')
	parser.add_argument('-user_thr', action='store_true', help='Ask for maximum threshold and step just after training')
	parser.add_argument('-display_curves', action='store_true', help='Display losses curves and Roc curve after training and scoring.')
	parser.add_argument('name', type=str, help='Name of the model')
	args = parser.parse_args()

	model_name = args.name

	# Fix random seed for reproducibility
	np.random.seed(7)

	if args.feature_select is True:
		# Apply feature selection to get best features
		bf = feature_selection.get_best_features(dataset_path, k_best, min_feature_importance)
	else:
		bf = None


	# Load data and get shape 
	train_files, test_files_n, test_files_a = utils.get_dataset_files(dataset_path)
	data_shape = get_data_shape(train_files)


	#--- Train seq2seq model 

	model = seq2seq(model_name, data_shape, hidden_dim, layers_stacked_count, learning_rate, lr_decay, momentum)

	train_start = time.time()
	train_loss, valid_loss, anomal_loss = model.train(train_file_list=train_files, 
													validation_file_list=test_files_n,
													anomalous_file_list=test_files_a, 
													best_features=bf, 
													epoch=epoch)
	training_time = time.time() - train_start

	
	if args.display_curves:
		#--- Plot loss over time:
		
		plt.figure(figsize=(12, 6))
		plt.plot(np.array(range(0, len(train_loss))) / float(len(train_loss) - 1) * (len(train_loss) - 1),
																									    np.log(train_loss),
																									    label="Train set loss")
		if valid_loss:
			plt.plot(np.log(valid_loss), label="Validation set loss")

		if anomal_loss:
			plt.plot(np.log(anomal_loss), label="Anomalous set loss")

		plt.title("Training errors over time (on a logarithmic scale)")
		plt.xlabel('Iteration')
		plt.ylabel('log(Loss)')
		plt.legend(loc='best')
		plt.show()
	


	#--- Compute ROC Curve 
	if args.user_thr:
		end_thr = int(input('[ ? ] Choose the max threshold for ROC curve : '))
		step_thr= int(input('[ ? ] Choose a threshold step : '))


	false_positives, true_positives = model.get_roc(strt_thr=start_thr, 
													end_thr=end_thr, 
													step=step_thr, 
													files_test_normal=test_files_n, 
													files_test_anomalous=test_files_a, 
													best_features=bf)
	
	write_summary(train_loss, valid_loss, anomal_loss, start_thr, end_thr, false_positives, true_positives,training_time, bf, data_shape[2])

	print('[DONE]')