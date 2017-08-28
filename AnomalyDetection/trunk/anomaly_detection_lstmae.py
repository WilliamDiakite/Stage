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
from class_LSTMAutoencoder import LSTMAutoencoder 
from seq2seq_data_preparation import get_seq2seq_batch



# Path to "normal" dataset
dataset_path 	= './../../Ressources/Generated_files/Datasets/big'

#------ Hyperparameters ------#

# Architecture
hidden_dim = 30

# Learning parameters
learning_rate = 0.0007

epoch = 400
seq_length = 20
batch_size = 10
#-----------------------------#


#--- Detection parameters  ---#

start_thr = 0
end_thr = 200
step_thr = 40
#-----------------------------#


#--- Feature selection ---#
k_best = 5
min_feature_importance = 0.5
#-------------------------#



def get_data_shape(file_list, best_features):
	
	data = pd.read_csv(file_list[0])

	if best_features is not None:
		data = data[best_features].values
	else:
		data = data.values

	data = get_seq2seq_batch(data, seq_length, batch_size)
	return data.shape


def write_summary(train_loss, valid_loss, anomal_loss, 
					start_thr, end_thr, false_positives, true_positives,
					training_time, best_features):
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
	parameters.write('Batch size : {}\n'.format(batch_size))
	parameters.write('Epochs : {}\n'.format(epoch))

	parameters.write('\nOptimizer : RMSprop\n')
	parameters.write('Learning rate : {}\n'.format(learning_rate))

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
	data_shape = get_data_shape(train_files, bf)
	print(data_shape)


	#--- Train seq2seq model 

	model = LSTMAutoencoder(model_name, data_shape, hidden_dim, learning_rate)

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
		final = 'n'
		while(final != 'y') :
			end_thr = float(input('[ ? ] Choose the max threshold for ROC curve : '))
			step_thr= float(input('[ ? ] Choose a threshold step : '))
			nb_points = int(end_thr / step_thr)
			
			print('[ ? ] ROC will have {} points.'.format(nb_points))
			
			final = input('[ ? ] Are you sure you want to keep these settings ? [y/n]')
			final = final.lower()


	false_positives, true_positives = model.get_roc(strt_thr=start_thr, 
													end_thr=end_thr, 
													step=step_thr, 
													files_test_normal=test_files_n, 
													files_test_anomalous=test_files_a, 
													best_features=bf)
	
	write_summary(train_loss, valid_loss, anomal_loss, start_thr, end_thr, false_positives, true_positives,training_time, bf)

	print('[DONE]')