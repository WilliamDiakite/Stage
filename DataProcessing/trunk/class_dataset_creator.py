# Creates a dataset from multiple csv files

import numpy as np
import csv
import os
import operator
import math

import warnings

from math import floor
from itertools import islice, tee
from scipy.interpolate import UnivariateSpline, InterpolatedUnivariateSpline, interp1d, LSQUnivariateSpline
from collections import OrderedDict




#-------------------------------------------------------------------------------#
# 							OUT OF THE CLASS METHODS 							#
#-------------------------------------------------------------------------------#

def load_raw_data(sensors, path_to_csv):
	'''
		Création du dictionnaire,
		parcours du fichier :
		pour chaque ligne on extrait les 3 paramètres de la ligne 	
		Si le nom du capteur est une key du dictionnaire, on ajoute (temps, valeur_capteur)
		sinon on ajoute la clé et le tuple (temps, valeur_capteur)
	'''
	print()
	print('[...] Processing', path_to_csv)

	data = {}

	with open(path_to_csv, newline='') as f:
		reader = csv.reader((line.replace('\0','') for line in f), delimiter=',')
		for row in [x for x in reader if len(x)==3 and x[1] in sensors]:
			
			# Update label name
			label = 'raw_' + row[1]
				
			if label in data:
				try:
					data[label].append((floor(float(row[0])), float(row[2])))
				except:
					pass
			else:
				try:
					data[label] = [(floor(float(row[0])), float(row[2]))]
				except:
					pass

	#print('[ + ] LABELS IN DATA:', list(data.keys()))
	data = OrderedDict(sorted(data.items()))

	return data

def remove_empty_zone(dataset, period):

	# Create copy that will be modified

	dataset_copy = dataset.copy()

	for label in dataset:
		
		dates = np.asarray([x[0] for x in dataset[label]])
		diff_dates = np.diff(dates)
		target_index = np.where(diff_dates > period)


		# Remove empty zone if needed
		if target_index[0].size > 0:
			
			for idx in target_index[0]:
				borne_min = dates[idx]
				borne_max = dates[idx+1]

				for label in dataset:
					tmp_serie = dataset_copy[label]
					dataset_copy[label] = [x for x in tmp_serie if x[0] > borne_max or x[0] < borne_min]

	return dataset_copy

def clean_data(dataset, period):

	print('\t[ + ] Cleaning data : removing empty chunks that are more than', period, 'seconds long')
	
	new_data = remove_empty_zone(dataset, period)
	old_data = new_data.copy()
	ite = 0

	while(True)	:
		
		ite += 1	
		new_data = remove_empty_zone(new_data, period)
		infos = compute_data_loss(new_data, dataset, period)
		
		if new_data == old_data:
			break
		else:
			old_data = new_data.copy()

	# Display average loss
	infos = compute_data_loss(new_data, dataset, period)
	losses = [l[2] for l in infos]
	avg_loss = sum(losses)/len(losses)
	print('\t[ + ] Average loss after cleaning file :', '%.2f' % avg_loss, '%')
	'''
	for item in infos:
		print(item[0], 'has', item[1].size, 'empty chunks remaining. Data loss :', item[2],'%')
	'''
	return new_data
	
def compute_data_loss(cleaned_data, original_data, period):
	
	infos = []

	# Verify how many empty zones are left
	for label in cleaned_data:
		dates = np.asarray([x[0] for x in cleaned_data[label]])
		diff_dates = np.diff(dates)
		
		# Start indexes where chunk is empty during more than period
		target_index = np.where(diff_dates > period)

		# Compute data loss
		loss_prct = ((len(original_data[label]) - len(cleaned_data[label]))*100)/len(original_data[label])

		# Store infos
		infos.append((label, target_index[0], loss_prct))

	'''
	# Display infos
	for item in infos:
		print(item[0], 'has', item[1].size, 'empty chunks remaining. Data loss :', item[2],'%')
	'''

	return infos

def get_min_max_date(data):
	'''
	 Returns min and max time from a dict dataset
	'''
	# Get min and max dates
	min_date = 10000000000000
	max_date = 0
	for label in data:
		tmp_time = [x[0] for x in data[label]]

		if max_date < max(tmp_time):
			max_date = max(tmp_time)

		if min_date > min(tmp_time):
			min_date = min(tmp_time)
	'''
	max_date -= min_date
	min_date -= min_date
	'''
	return min_date, max_date

def window(it, size=3):
	'''
		Split an array in iterable array of size n

		Parameters :
			it : array to split
			size : size of the window
		Return :
			Iterable object that contains all the windows
	'''
	yield from zip(*[islice(it, s, None) for s, it in enumerate(tee(it, size))])

def moving_average(values, window_size):
	'''
		Compute moving average for an array of values using
		a window

		Parameters:
			values : orginal array
			window_size : size of the window
		Return :
			sma : moving average array
	'''
	weights = np.repeat(1.0, window_size)/window_size
	sma = np.convolve(values, weights, 'same')
	return sma



#-------------------------------------------------------------------------------#
# 						========= 	CLASS 	=========							#
#-------------------------------------------------------------------------------#

class DatasetCreator(object):
	"""
		Loads csv file
		Creates raw dataset and compute complete dataset
	"""
	def __init__(self, dataset_name, out_data, data=None, labels={}):
		
		self._name 	 = dataset_name
		self.out_data= out_data
		
		self._data 	 = data
		self._labels = labels
		self._splines= {}

		self._duration = 0

	@property
	def name(self):
		return self._name

	@property
	def shape(self):
		return self._data.shape

	@property
	def labels(self):
		return self._labels

	@property
	def n_sample(self):
		return self._data.shape[1]

	@property 
	def n_features(self):
		return self._data.shape[0]-1

	@property
	def sensors(self):
		return list(self._labels.keys())

	@property
	def is_empty(self):
		if self._data is None:
			return True
		else: 
			return False

	#-------------------------------------------------------------------------------#
	# 							DATA INITIALIZATION		 							#
	#-------------------------------------------------------------------------------#
	
	def init_dataset_from(self, sensors, path):
		"""
			Initialisation du dataset
				Lit tous les fichier csv, 
				Créer une table 'raw'
				Calcule les interpolations,
				finalise le dataset (sans features) 
		"""
		# Récupère les données du fichier sous forme de dictionnaire
		data = load_raw_data(sensors, path)

		# Retire les sections vides
		data = clean_data(data, 240)

		# If data is empty return None
		if not data:
			self._data = None
			return None

		# If one of the sensors array is empty return None
		for sensor in data:
			if len(data[sensor]) < 1:
				self._data = None
				return None

		# Init labels dict
		self._labels = {}
		l = list(data.keys())
		l.insert(0, 'Time')
		for label in l:
			self._labels[label] = l.index(label)

		# Construct raw table (with nan) and save
		self.construct_table(data)

		# Test raw matrix shape to check if there's the right nb of sensors
		if self._data.shape[0]-1 != len(sensors):
			print('[ ! ] Error during dataset init')
			print('[-->] Data should have', len(sensors), 'sensors but has', self.n_features)
			return None

		# Compute spline for all time series
		self.compute_splines(data)

		# Init dataset with approximated values
		self.build_dataset(data)

	def construct_table(self, data):
		
		min_date, max_date = get_min_max_date(data)
		
		# Update max date for file
		self._duration = max_date - min_date
		result = np.ones(shape=(len(data)+1, max_date-min_date))
		result.fill(np.nan)
		result[0] = np.arange(max_date-min_date)
				
		# Remplissage de la table résultante
		for label in data:
			lig = self._labels[label]
			for (t, value) in data[label]:
				t -= min_date
				try:
					result[lig][t] = value
				except IndexError:
					pass
		self._data = result
		'''
		for i in range(self._data.shape[0]):
			print('Result :', self._data[i])
		'''
		# Export to csv for raw visualization
		filename = self.out_data + '/raw/'+self._name
		self.to_csv(file=filename)
		
	def compute_splines(self, data):
		'''
			Calcul et sauve les fonctions d'interpolation
			grace au dictionnaire renvoyé par clean_data(...)
		'''
		self._splines = {}

		min_date, _ = get_min_max_date(data)
		
		for sensor in data:
			values = [x[1] for x in data[sensor]]
			dates  = [x[0]-min_date for x in data[sensor]]

			# Start and end time series with 0 values
			# so approximated points are not out of bounds
			dates.insert(0,0)
			values.insert(0,-1)
			dates.append(self._duration+1)
			values.append(0)

			spl = interp1d(dates, values, kind='nearest', bounds_error=False)

			self._splines[sensor] = spl
				
	def build_dataset(self, data):

		min_date, max_date = get_min_max_date(data)
		
		result = np.ones(shape=(len(self._splines)+1, max_date-min_date))
		result.fill(np.nan)
		result[0] = np.arange(max_date-min_date)	
		
		# Computing interpolated/approximated values
		for sensor in self._splines:
			result[self._labels[sensor]] = self._splines[sensor](result[0])		
	
		# Updating self 
		self._data = result
		
	def to_csv(self, file):
		'''
			Save dataset to csv
		'''
		# Sort labels by sensor index
		labels = ','.join([x[0] for x in sorted(self._labels.items(), key=operator.itemgetter(1))])
		np.savetxt(file, self._data.T, header=labels, delimiter=',', comments='')
		

	#-------------------------------------------------------------------------------#
	# 						DATA INTERPOLATION & CLEANING							#
	#-------------------------------------------------------------------------------#


	def nan_checker(self):
		if np.isnan(self._data).any(): 
			print('[ ! ] NANs detected at sensor')


	#-------------------------------------------------------------------------------#
	# 							FEATURES CREATION 									#
	#-------------------------------------------------------------------------------#
	'''
	def add_derivatives(self):
		"""
			Add 1st and 2nd derivative
			Using splines
		"""
		for label in self._splines: 
			#--- Get 1st and 2nd derivative
			d1 = self._splines[label].derivative(n=1)
			d2 = self._splines[label].derivative(n=2)

			new1 = d1(self._data[0])
			new2 = d2(self._data[0])

			#-- removing nans
			nans1 = np.isnan(new1)
			nans2 = np.isnan(new2)
			new1[nans1] = 0
			new2[nans2] = 0 
			
			#--- Add 1st derivative to self._data		
			self._data = np.vstack((self._data, new1))
			
			#--- Add 2st derivative to self._data
			self._data = np.vstack((self._data, new2))
			
			#Append labels
			l = 'd1_'+label[4:]
			self._labels[l] = max(self._labels.values())+1
			l = 'd2_'+label[4:]
			self._labels[l] = max(self._labels.values())+1
	'''

	def add_derivatives(self):
		'''
			Add 1st and 2nd derivative
			Using linear interpolation
		'''
		for label in self._splines:
			d1 = np.diff(self._data[self._labels[label]]) / np.diff(self._data[0])
			d1 = np.insert(d1, 0, 0)
			
			d2 = np.diff(d1) / np.diff(self._data[0])
			d2 = np.insert(d2, 0, 0)

			#-- removing nans
			nans1 = np.isnan(d1)
			nans2 = np.isnan(d2)
			d1[nans1] = 0
			d2[nans2] = 0 

			# Update self._data and self._labels
			l = 'd1_'+label[4:]
			self._labels[l] = max(self._labels.values())+1
			self._data = np.vstack((self._data, d1))

			l = 'd2_'+label[4:]
			self._labels[l] = max(self._labels.values())+1
			self._data = np.vstack((self._data, d2))

	def add_std(self, windows):
		'''
			Ajout de l'ecart-type sur la fenêtre d'étude pour 
			chaque capteur
		'''
		_, length = self._data.shape
		
		for label in [x for x in self._labels if x[0]=='r']:
			
			for n in windows:
				tmp = np.ones(length)
				start = floor(n/2)
				
				# For every window, compute std
				for l in window(self._data[self._labels[label]], n):
					tmp[start] = np.std(l)
					start = start+1
					
				# Update dataset with new feature	
				self._data = np.vstack((self._data, tmp))
					
				# Append label	
				l = 'std_w'+str(n)+'_'+label[4:]
				self._labels[l] = max(self._labels.values())+1

	def add_moving_average(self, windows):
		'''
			Ajout de la moyenne glissante sur n secondes
			Parameters :
				windows : taille des fenetres
				append_label : ajout du label si necessaire
		'''
		_, length = self._data.shape

		# For every sensor
		for label in [x for x in self._labels if x[0]=='r']:
			
			# Compute moving average for every window size
			for w in windows:
				mva = moving_average(self._data[self._labels[label]], window_size=w)
				tmp = np.zeros(length)
				tmp = mva
				try:
					self._data = np.vstack((self._data, tmp))
				except:
					pass

				# Append labels
				l = 'ma_'+str(w)+'_'+label[4:]
				self._labels[l] = max(self._labels.values())+1
		

	#-------------------------------------------------------------------------------#
	# 							ADD ALL FEATURES 									#
	#-------------------------------------------------------------------------------#	

	def add_all_features(self, std_win, mv_avg_win):
		
		# Check for nans
		self.nan_checker()		

		# Add all features features
		self.add_moving_average(windows=mv_avg_win)
		self.add_std(windows=std_win)
		self.add_derivatives()
		
				