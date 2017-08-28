'''
							-------- HOW TO ------- 

-- Etape 1 : 
Ouvrir le fichier dataset_creator.py dans un éditeur de texte. 

-- Etape 2 (modification des chemins de données):
Modifier les chemins raw_normal_path et raw_anomalous_path vers les 
dossier contant les fichiers .csv de données normales et anormales brutes.
Modifier le chemin de destination (out_path) et le nom des deux datasets (normal_name, anomalous_name).

-- Etape 3 (sélection des capteurs):
Les fichiers .csv de données brutes fournis dispose d'information sur plusieurs capteurs. 
Il est possible de choisir les capteurs à enregistrer et traiter. Pour ce faire, modifier 
le paramètre sensors pour y ajouter ou enlever des capteurs. 
Le nom du capteur doit être exacte (liste complète des capteurs disponibles en annexes).

-- Etape 4 (paramètres d'ajout de features) :
Modifier les tailles de fenêtres pour l’ajout de moyennes glissantes (mv_avg_win) et écarts-types (std_win).
Exemple :
	si  mv_avg_win = [7, 13, 27], 3 features (ici, moyennes glissantes) seront rajoutées. 
	Le premier calcul de moyenne glissante sera fait sur 7 secondes, le second sur 13 secondes 
	et le troisième sur 27 secondes.
	Le fonctionnement est identique pour l'ajout d’écarts-types (avec std_win)
Le nombre de fenêtres n'est pas limité.


-- Etape 5 (exécution) :
Dans un terminal exécuter le script python :
	> python3 dataset_creator.py
Vérifier que les deux jeux données ont été créés dans le dossier de destination out_path.

'''


from utils import build_dataset

#-------------------------------------------------------------------------------#
# 							APPLICATION PARAMETERS 								#
#-------------------------------------------------------------------------------#
																				
# Path to raw dataset 															
raw_normal_path 	= './../../Ressources/Raw_data/normal/'										
raw_anomalous_path 	= './../../Ressources/Raw_data/not_normal/'

# Datasets destination
out_path = './../../Ressources/Generated_files/Datasets/'

# Dataset name
dataset_name = 'big'

# Sensor list
#sensors = ['ACCELERATOR_POS_D', 'ACCELERATOR_POS_E', 'FUEL_INJECT_TIMING',
#			'AMBIANT_AIR_TEMP', 'RPM', 'SPEED', 'THROTTLE_POS', 'THROTTLE_ACTUATOR']

#sensors = ['SPEED', 'RPM', 'INTAKE_PRESSURE']

sensors = ['ACCELERATOR_POS_D', 'ACCELERATOR_POS_E', 'AMBIANT_AIR_TEMP', 'CATALYST_TEMP_B1S1',   
'COMMANDED_EGR', 'COOLANT_TEMP', 'EGR_ERROR', 'ENGINE_LOAD', 'FUEL_INJECT_TIMING',  
'FUEL_RAIL_PRESSURE_DIRECT', 'INTAKE_PRESSURE', 'INTAKE_TEMP', 'MAF', 
'THROTTLE_ACTUATOR', 'THROTTLE_POS', 'RPM', 'SPEED']

#sensors = ['ENGINE_LOAD', 'INTAKE_PRESSURE', 'MAF', 'RPM', 'FUEL_INJECT_TIMING']


# Window sizes in seconds(standard deviation, moving average)
# example: [51, 101, 303]
std_win 	= [12, 60]															
mv_avg_win 	= [12, 60]																																		


#-------------------------------------------------------------------------------#
# 						LOAD RAW DATA - BUILD DATASETS 							#
#-------------------------------------------------------------------------------#

out_path = out_path + dataset_name + '/'

#--- Build normal dataset
build_dataset(rdata_path=raw_normal_path,sensors=sensors, name='normal',
				std_win=std_win, mv_avg_win=mv_avg_win, out=out_path)

#--- Build anomalous dataset
build_dataset(rdata_path=raw_anomalous_path,sensors=sensors, name='anomalous',
				std_win=std_win, mv_avg_win=mv_avg_win, out=out_path)
