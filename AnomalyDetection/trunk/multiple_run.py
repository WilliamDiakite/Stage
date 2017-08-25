# coding: utf8

# Anomaly detection mutiple run
import os
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('name', type=str, help='Choose a model base name')
parser.add_argument('model_type', type=str, choices=['autoencoder', 'lstm', 'lstmae'], help='Choose a model to run')
parser.add_argument('nb', type=int, help='Positive number that determines how train_score will be run')
args = parser.parse_args()

for i in range(args.nb):

	if args.model_type == 'autoencoder':
		cmd = 'python3 anomaly_detection.py {}_{}'.format(args.name, str(i))
	if args.model_type == 'lstm':
		cmd = 'python3 anomaly_detection_lstm.py {}_{}'.format(args.name, str(i))
	if args.model_type == 'lstmae':
		cmd = 'python3 anomaly_detection_lstmae.py {}_{}'.format(args.name, str(i))
	
	os.system(cmd)