from argparse import ArgumentParser
import os
import time

parser = ArgumentParser()
parser.add_argument('dataset', type=str, default=None, help="Dataset name")
parser.add_argument('normality', type=str, default='normal', help='Choose normal or anomalous data', choices=['normal', 'anomalous'])
parser.add_argument('file', type=str, default=None, help="file number")
parser.add_argument('idx', type=str, default=None, help="Column's index to plot")
parser.add_argument('-norm', action='store_true', help='Plot normalized data')
parser.add_argument('-raw', action='store_true', help='Plot raw data')
parser.add_argument('-pro', action='store_true', help='Plot processed data')
args = parser.parse_args()




#-------------------------------------------------------------------------------------------------------#
#  This file WILL NOT be executed. After every modification, please make sure you copy this file in :	#		
#  --->	/PredictiveCarMaintenance/Ressources/Generated_files/Datasets/									#
# and also update cmd_in path																			#
#-------------------------------------------------------------------------------------------------------#



#-------------------------

def plot(name, idx):
	if name == 'raw':
		return name + ' u 1:' + idx
	else:
		return name + ' u 1:' + idx + ' with line'

#------------------------


commands = open("gnuplot_in", 'w')
print('set datafile separator ","', file=commands)

plot_cmd = 'plot '

nb_plot = 0

if args.norm:
	# Set variable path
	path = args.dataset + '/' + args.normality + '/' + 'normalized' + '/' + args.file
	cmd = 'norm = ' + '"' + path + '"'
	print(cmd, file=commands)

	# Update plot command
	if nb_plot == 0:
		plot_cmd += plot('norm', args.idx)
		nb_plot += 1
	else:
		plot_cmd += ' , ' + plot('norm', args.idx)


if args.raw:
	# Set variable path
	path = args.dataset + '/' + args.normality + '/' + 'raw' + '/' + args.file
	cmd = 'raw = ' + '"' + path + '"'
	print(cmd, file=commands)

	# Update plot command
	if nb_plot == 0:
		plot_cmd += plot('raw', args.idx)
		nb_plot += 1
	else:
		plot_cmd += ' , ' + plot('raw', args.idx)


if args.pro:
	# Set variable path
	path = args.dataset + '/' + args.normality + '/' + 'not_normalized' + '/' + args.file
	cmd = 'norm = ' + '"' + path + '"'
	print(cmd, file=commands)

	# Update plot command
	if nb_plot == 0:
		plot_cmd += plot('norm', args.idx)
		nb_plot += 1
	else:
		plot_cmd += ' , ' + plot('norm', args.idx)


print(plot_cmd, file=commands)
print('pause -1', file=commands)

commands = open('gnuplot_in', 'r+')
os.system('gnuplot -persist gnuplot_in')

# Remove command
os.remove('gnuplot_in')
