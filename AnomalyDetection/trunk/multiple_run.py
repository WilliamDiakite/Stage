# coding: utf8

# Anomaly detection mutiple run
import os
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('nb', type=int, help='Positive number that determines how train_score will be run')
args = parser.parse_args()

for i in range(args.nb):
	cmd = 'python3 anomaly_detection.py train_score model' + str(i) + ' -force_reset'
	os.system(cmd)