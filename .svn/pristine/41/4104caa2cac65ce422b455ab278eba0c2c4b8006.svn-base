from class_dataset_creator import DatasetCreator
import os, shutil

import numpy as np

import pandas as pd
from pandas import read_csv

from sklearn.preprocessing import MinMaxScaler, StandardScaler


def clean_dir(path):
	for the_file in os.listdir(path):
	    file_path = os.path.join(path, the_file)
	    os.remove(file_path)


def check_create_dir(dir_name):
	if not os.path.isdir(dir_name):
		#try:
		os.mkdir(dir_name)
		'''
		except OSError:
			print('[ ! ]', dir_name, 'hasnt been created !')
			exit('[-->] Exception OSError')
		'''


def build_dataset(rdata_path, sensors, name, std_win, mv_avg_win, out):
	'''
		Go through all the raw file and build full dataset
		Also export raw dataset for visualization and features validation

		Parameters:
			rdata_path 	: (str) path to raw data folder
			sensors 	: (list<str>) list of sensors to process
			name 		: (str) name of the dataset
			std_win 	: (list<int>) window sizes for standard deviation features
			mv_avg_win 	: (list<int>) window sizes for moving average features
	'''
	file_counter = 1

	# Creating subfodlers
	data_dir = out + name 
	raw_dir = data_dir + '/raw/'
	norm_dir = data_dir + '/normalized/'
	not_norm_dir = data_dir + '/not_normalized/'

	check_create_dir(out)
	check_create_dir(data_dir)
	check_create_dir(raw_dir)
	check_create_dir(norm_dir)
	check_create_dir(not_norm_dir)

	n_features = len(sensors) * (3 + len(mv_avg_win) + len(std_win))

	# Remove all old files inside dir
	clean_dir(not_norm_dir)
	clean_dir(norm_dir)

	# Reading all csv files from root directory
	for subdir, dirs, files in os.walk(rdata_path):
				
		for file in [x for x in files if x == 'csv.csv']:
		
			tmp_file_name = str(file_counter) + '.csv'
			d = DatasetCreator(dataset_name=tmp_file_name, out_data=data_dir)
			d.init_dataset_from(sensors=sensors, path=os.path.join(subdir, file))
			
			if d.is_empty:
				print('[ ! ] Skipping cause empty data :', file)
				break

			# If number of sensor found is different from sensor list
			if d.n_features != len(sensors):
				print('[ ! ] Skipping cause not enough sensors in file :', file)
				break
			else:
				# Adding all features and adding data to csv file f
				d.add_all_features(std_win=std_win, mv_avg_win=mv_avg_win)
				if d.n_features != n_features:
					print('[ + ] Problem adding features, skipping file')
				else:	
					filename = d.out_data + '/not_normalized/' + d.name
					d.to_csv(file=filename)
					file_counter += 1
						
	print('[ + ] Dataset',name,'has been saved and is composed of', n_features, 'features')


def normalize_all_files(path):
	'''
		Normalize all files separately.
		Fit the scaler using 60% of normal data 
	'''
	path_normal 	= path + '/normal/not_normalized/'
	path_anomalous 	= path + '/anomalous/not_normalized/'
	
	path_normal_norm	= path + '/normal/normalized/'
	path_anomalous_norm	= path + '/anomalous/normalized/'

	clean_dir(path_normal_norm)
	clean_dir(path_anomalous_norm)


	if not os.path.isdir(path_normal) or not os.path.isdir(path_anomalous):
		exit('[ ! ] Nothing found at', path)

					
	# Normalize all normal files separately
	for filename in os.listdir(path_normal):

		# Init a scaler for every file
		sc = StandardScaler()
		
		in_file= path_normal + filename
		out_file= path_normal_norm + filename

		# Scale data
		try:
			tmp = read_csv(in_file, index_col=False).values[:, 1:]
			tmp = sc.fit_transform(tmp)
			norm = np.vstack((np.arange(tmp.shape[0]), tmp.T)).T
			np.savetxt(out_file, norm, delimiter=',')
		except:
			print('shape :', shape)
			print(tmp.shape)
			print('[ ! ] Problem scaling :')
			print('[-->]', out_file)
	
	# Normalize all anomalous files separately
	for filename in os.listdir(path_anomalous):

		# Init a scaler for every file
		sc = StandardScaler()

		in_file= path_anomalous + filename
		out_file = path_anomalous_norm + filename
		
		# Scale data
		try:
			tmp = read_csv(in_file, index_col=False).values[:, 1:]
			tmp = sc.fit_transform(tmp)
			norm = np.vstack((np.arange(tmp.shape[0]), tmp.T)).T
			np.savetxt(out_file, norm, delimiter=',')
		except:
			print('[ ! ] Problem scaling :')
			print('[-->]', out_file)
			print('[-->] File not saved')