# coding: utf8

import os
import csv

import utils

import numpy as np
from pandas import read_csv

from sklearn.feature_selection import RFE, SelectKBest, chi2, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler


from beautifultable import BeautifulTable


nb_features_to_keep = 40
min_feature_importance = 0.06 #  0 > min_feature_importance >= 0.1

#---------------------------------------------------#
#						Utils						#
#---------------------------------------------------#

def clean_dir(path):
	for the_file in os.listdir(path):
	    file_path = os.path.join(path, the_file)
	    os.remove(file_path)

def get_dataset_files_fs(dataset_path):
	'''
		Scales ALL data for feature selection
		Scaled files are saved in Normalized/ normal and anomalous folders
	'''
	print('\t[...] Retrieving datasets')

	# Get paths to data
	path_normal		= dataset_path + '/normal/not_normalized/'
	path_anomalous	= dataset_path + '/anomalous/not_normalized/'

	dest_normal		= dataset_path + '/normal/normalized/'
	dest_anomalous	= dataset_path + '/anomalous/normalized/'

	# Removes every normalized files if any
	clean_dir(dest_normal)
	clean_dir(dest_anomalous)
	
	# Get list of files
	normal_files = [path_normal+x for x in os.listdir(path_normal)]
	anomal_files = [path_anomalous+x for x in os.listdir(path_anomalous)]
	all_files = normal_files + anomal_files
	
	# Scale data
	print('\t[...] Fitting scaler')
	scaler = utils.fit_scaler(all_files)

	# Apply std and save files @ dest_...
	print('\t[...] Scaling data')
	utils.standardize(normal_files, scaler, dest_normal)
	utils.standardize(anomal_files, scaler, dest_anomalous)

def check_create_dir(dir_name):
	'''
		Creates a directory if dir_name doesn't exists

		Argument:
			dir_name : name of directory to check/create
	'''
	if not os.path.isdir(dir_name):
		os.mkdir(dir_name)

def create_single_set(file_list, label):
	'''
		USED ONLY FOR FEATURE SELECTION
		Create a unique dataframe from all files

		Arguments 
			file_list : list of the file to append
			label : 0 or 1 (normal or anomalous)

		Return 
			data : complete dataset
			labels : array of label (0 or 1)
	'''
	data = None

	for file in file_list:
		
		if data is None:
			data = read_csv(file, index_col=False).values[1:]
		else:
			tmp = read_csv(file, index_col=False).values[1:]
			data = np.vstack((data, tmp))

	# Add labels (0: normal, 1: anomalous)
	labels = np.ones(data.shape[0]) * label
	return data, labels

def get_file_list(dataset_path):
	'''
		Get normal and anomalous full filenames 

		Argument 
			dataset_path : path to dataset

		Returns
			normal_list : list of all normal files
			anomal_list : list of all anomalous files
	'''
	# Set data path
	normal_path	= dataset_path + '/normal/normalized/'
	anomal_path	= dataset_path + '/anomalous/normalized/'

	# Get file lists
	normal_list = [normal_path + x for x in os.listdir(normal_path)]
	anomal_list = [anomal_path + x for x in os.listdir(anomal_path)]
	
	# Use only half of the dataset to select features
	normal_list = normal_list[:int(len(normal_list)/2)]
	anomal_list = anomal_list[:int(len(anomal_list)/2)]

	return normal_list, anomal_list

def init_data(dataset_path):

	# Get normal files and anomalous files
	normal_list, anomal_list = get_file_list(dataset_path)

	# Create unique tables for normal and anomalous data
	normal_data, normal_class = create_single_set(normal_list, label=0)
	anomal_data, anomal_class = create_single_set(anomal_list, label=1)

	# Stack for feature selection
	X = np.vstack((normal_data, anomal_data))
	Y = np.hstack((normal_class, anomal_class))

	return X, Y

def get_labels(dataset_path):
	'''
		Select a random file in not normalized folder
		in order to get labels
	'''
	file_path = dataset_path + '/normal/not_normalized/1.csv'

	try:
		df = read_csv(file_path, index_col=False)
	except:
		exit('[ ! ] /not_normalized/1.csv does not exist')

	return np.asarray(list(df)[1:])
		
def display_feature_importance(labels, importance_result):
	table = BeautifulTable()
	table.column_headers = ["Sensor Name", "Feature Importance"]
	data = list(zip(labels[1:], importance_result))
	data.sort(key=lambda tup: tup[1], reverse=True)
	for e in data: 
		table.append_row([e[0], e[1]])
	print(table)

def display_rfe_ranking(labels, rfe_rank):
	table = BeautifulTable()
	table.column_headers = ["Sensor Name", "Rank (lower is better)"]
	data = list(zip(labels, rfe_rank))
	data.sort(key=lambda tup: tup[1])
	for e in data: 
		table.append_row([e[0], e[1]])
	print(table)




#---------------------------------------------------#
#			Feature selection algorithms			#
#---------------------------------------------------#


def recursive_feature_elimination(dataset_path, nb_features_to_keep):

	print('\t[-->] Applying recursive feature elimination ')

	# Load data
	data, labels = init_data(dataset_path)

	# Apply RFE
	model = LogisticRegression()
	rfe = RFE(model, nb_features_to_keep)
	fit = rfe.fit(data[:,1:], labels)

	# Display selected features
	labels = get_labels(dataset_path)[1:]

	
	selected_idx = np.where(np.asarray(fit.support_) == True)
	#to_drop_idx  = np.where(np.asarray(fit.support_) == False)
	
	display_rfe_ranking(labels, fit.ranking_)
	
	return labels[selected_idx]


def feature_importance(dataset_path, min_feature_importance):

	print('\t[-->] Estimating feature importance')

	# Load data
	data, data_class = init_data(dataset_path)

	# Estimate feature importance
	model = ExtraTreesClassifier()
	model.fit(data[:,1:], data_class)
	
	# Display selected features
	labels = get_labels(dataset_path)
	
	print('\t[info] OK features have importance > ', min_feature_importance)
	display_feature_importance(labels, model.feature_importances_)
	
	
	selected_idx = np.where(np.asarray(model.feature_importances_) > min_feature_importance)
	#to_drop_idx = np.where(np.asarray(model.feature_importances_) <= 0.05)
	
	return labels[selected_idx]


def univariate_selection(dataset_path):
	'''
		Ne fonctionne pas pour le moment
	'''

	print('[...] Applying univariate selection')

	# Load data
	data, labels = init_data(dataset_path)

	# feature extraction
	test = SelectKBest(f_classif, k=7)
	fit = test.fit(data[:,1:], labels)

	# Display selected features
	labels = get_labels(dataset_path)
	selected = np.where(np.asarray(fit.scores_) > 200)
	print('\t[-->] Selected Features:', labels[selected])


def get_best_features(dataset_path, k_best, min_feature_importance):

	print('\n[ + ] Applying feature selection')

	# Normalize all data
	get_dataset_files_fs(dataset_path)

	# Apply feature selection and keep best features labels
	ref_labels = recursive_feature_elimination(dataset_path, k_best)
	fi_labels = feature_importance(dataset_path, min_feature_importance)

	# Remove duplicates
	best_features = list(ref_labels) + list(fi_labels)
	best_features = list(set(best_features))

	print('\t[-->] Best features :')
	for i in best_features:
		print('\t',i)

	return best_features


if __name__ == '__main__':

	dataset_path = './../../Ressources/Generated_files/Datasets/new_new'
	
	'''
	get_dataset_files_fs(dataset_path)

	recursive_feature_elimination(dataset_path)
	feature_importance(dataset_path)
	#univariate_selection(dataset_path)
	'''

	get_best_features(dataset_path)