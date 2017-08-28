# coding: utf8

import os
import sys
import time 

import numpy as np
import utils 
import tensorflow as tf
from beautifultable import BeautifulTable


import matplotlib.pyplot as plt
plt.style.use('ggplot')

from utils import DatasetInfo, drange
from class_Autoencoder import Autoencoder

from feature_selection import get_best_features

from pandas import read_csv

'''
Activation functions :
	tf.nn.relu(features, name=None)
	tf.nn.relu6(features, name=None)
	tf.nn.crelu(features, name=None)
	tf.nn.elu(features, name=None)
	tf.nn.softplus(features, name=None)
	tf.nn.softsign(features, name=None)
	tf.sigmoid(x, name=None)	(default)
	tf.tanh(x, name=None)


Optimizers :
	tf.train.Optimizer
	tf.train.GradientDescentOptimizer  (default)
	tf.train.AdadeltaOptimizer
	tf.train.AdagradOptimizer
	tf.train.AdagradDAOptimizer
	tf.train.MomentumOptimizer
	tf.train.AdamOptimizer     
	tf.train.FtrlOptimizer
	tf.train.ProximalGradientDescentOptimizer
	tf.train.ProximalAdagradOptimizer
	tf.train.RMSPropOptimizer
'''


#-------------------------------------------------------------------------------#
# 							APPLICATION'S PARAMETERS							#
#-------------------------------------------------------------------------------#

#--- DATA PATH ---#

# Path to "normal" dataset
dataset_path 	= './../../Ressources/Generated_files/Datasets/big'

# Stored models
stored_models = './Saved_models/'


#--- MODEL PARAMETERS ---#

# Model architecture (hidden layers only)
hidden_layers = [5, 2]

# Learning parameters
activation		= tf.nn.tanh
learning_rate 	= 0.001
optimizer		= tf.train.AdamOptimizer(learning_rate)
batch_size		= 100
epochs 			= 500
file_start 		= 0.0
file_end   		= 1


#--- Threshold parameters ---#

# plage de recherche du seuil
start_thr = 0
end_thr = 3

# pas de la recherche du seuil
step_thr = 0.05


#--- Feature selection ---#
k_best = 50
min_feature_importance = 0.5






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
	parameters.write('Model type : autoencoder\n')
	parameters.write('Dataset : {}\n'.format(dataset_path))

	parameters.write('\nInput dimension : {}\n'.format(io_dim))
	
	parameters.write('Hidden layers : {}\n'.format(hidden_layers))
	
	parameters.write('Output dimension : {}\n'.format(io_dim))

	parameters.write('\nBatch size : {}\n'.format(batch_size))
	parameters.write('Epochs : {}\n'.format(epochs))

	parameters.write('\nOptimizer : Adam\n')
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




#-------------------------------------------------------------------------------#
# 							TRAINING AND SCORING 								#
#-------------------------------------------------------------------------------#


if __name__ == '__main__':

	from argparse import ArgumentParser
	parser = ArgumentParser()
	parser.add_argument('-feature_select', action='store_true', help='Apply feature selection, only keep best features during training and testing')
	parser.add_argument('-user_thr', action='store_true', help='Ask for maximum threshold and step just after training')
	parser.add_argument('-display_curves', action='store_true', help='Display losses curves and Roc curve after training and scoring.')
	parser.add_argument('name', type=str, help='Name of the model')
	args = parser.parse_args()

	model_name = args.name

	#-------------------------------------------------------------------------------#
	#								APPLICATION INIT 								#
	#-------------------------------------------------------------------------------#
	
	if args.feature_select is True:
		# Apply feature selection to get best features
		bf = get_best_features(dataset_path, k_best, min_feature_importance)

		# Automatic architecture
		hidden_layers = []
		hidden_layers.append(int(len(bf)/2))
		hidden_layers.append(int(len(bf)/4))
		hidden_layers.append(int(len(bf)/10))
	else:
		bf = None

	


	# Initialize a data information collector
	di = DatasetInfo(dataset_path)
	#train_files, test_files_n, test_files_a = utils.get_dataset_files(dataset_path)
	train_files, test_files_n, test_files_a = utils.get_dataset_files(dataset_path)
	

	# Initialize model
	model = Autoencoder(model_name=args.name, 
						input_dim=di.nb_features() if bf is None else len(bf), 
						hidden_layers=hidden_layers, 
						optimizer=optimizer)


	#-----------------------------------------------------------------------------------#
	#								TRAINING & SCORING									#
	#-----------------------------------------------------------------------------------#

	
	#--- Training model

	print('[ + ] Training and testing model')

	train_start = time.time()
	train_loss, valid_loss, anomal_loss = model.train(train_files=train_files, 
														validation_files=test_files_n,
														anomalous_files=test_files_a,
														best_features=bf,
														training_epochs=epochs,
														batch_size=batch_size,  
														start_file_prct=file_start, 
														end_file_prct = file_end)
	training_time = time.time() - train_start

	#--- Plot loss curves 
	if args.display_curves:
		
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
		

	#--- Ask for ROC parameters
	if args.user_thr:
		final = 'n'
		while(final != 'y') :
			end_thr = float(input('[ ? ] Choose the max threshold for ROC curve : '))
			step_thr= float(input('[ ? ] Choose a threshold step : '))
			nb_points = int(end_thr / step_thr)
			
			print('[ ? ] ROC will have {} points.'.format(nb_points))
			
			final = input('[ ? ] Are you sure you want to keep these settings ? [y/n]')
			final = final.lower()


	# Compute roc curve
	false_positives, true_positives = model.get_roc(start_thr, end_thr, step_thr, 
														test_files_n, test_files_a, bf)
	
	write_summary(train_loss, valid_loss, anomal_loss, start_thr, end_thr, false_positives, true_positives,training_time, bf, di.nb_features())


	print('[DONE]')



def check_labels(labels):
	'''
		check for anomaly in label names
	'''
	for i in range(labels):
		if labels[i][:2] == '# ':
			labels[i] = labels[i][2:]
	return labels