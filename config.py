"""Configuration file for Knowledge Distillation experiments."""

import os
import numpy as np
# Dataset configuration
# UNIVARIATE_DATASET_NAMES_2018 = [
#     'ACSF1', 'Adiac', 'ArrowHead', 'Beef', 'BeetleFly', 'BirdChicken', 'BME', 'Car', 'CBF',
#     'Chinatown', 'ChlorineConcentration', 'CinCECGtorso', 'Coffee',
#     'Computers', 'CricketX', 'CricketY', 'CricketZ', 'Crop', 'DiatomSizeReduction',
#     'DistalPhalanxOutlineAgeGroup', 'DistalPhalanxOutlineCorrect', 'DistalPhalanxTW',
#     'Earthquakes', 'ECG200', 'ECG5000', 'ECGFiveDays', 'ElectricDevices', 'EOGHorizontalSignal',
#     'EOGVerticalSignal', 'EthanolLevel', 'FaceAll', 'FaceFour', 'FacesUCR', 'FiftyWords', 'FISH',
#     'FordA', 'FordB', 'FreezerRegularTrain', 'FreezerSmallTrain', 'GunPoint', 'GunPointAgeSpan',
#     'GunPointMaleVersusFemale', 'GunPointOldVersusYoung', 'Ham', 'HandOutlines', 'Haptics',
#     'Herring', 'HouseTwenty', 'InlineSkate', 'InsectEPGRegularTrain', 'InsectEPGSmallTrain',
#     'InsectWingbeatSound', 'ItalyPowerDemand', 'LargeKitchenAppliances', 'Lightning2', 'Lightning7',
#     'MALLAT', 'Meat', 'MedicalImages', 'MiddlePhalanxOutlineAgeGroup', 'MiddlePhalanxOutlineCorrect',
#     'MiddlePhalanxTW', 'MixedShapesSmallTrain', 'MoteStrain', 'NonInvasiveFetalECGThorax1',
#     'NonInvasiveFetalECGThorax2', 'OliveOil', 'OSULeaf', 'PhalangesOutlinesCorrect', 'Phoneme',
#     'PigAirwayPressure', 'PigArtPressure', 'PigCVP', 'Plane', 'PowerCons', 'ProximalPhalanxOutlineAgeGroup',
#     'ProximalPhalanxOutlineCorrect', 'ProximalPhalanxTW', 'RefrigerationDevices', 'Rock',
#     'ScreenType', 'SemgHandGenderCh2', 'SemgHandMovementCh2', 'SemgHandSubjectCh2', 'ShapeletSim',
#     'ShapesAll', 'SmallKitchenAppliances', 'SmoothSubspace', 'SonyAIBORobotSurface1',
#     'SonyAIBORobotSurface2', 'StarLightCurves', 'Strawberry', 'SwedishLeaf', 'Symbols',
#     'SyntheticControl', 'ToeSegmentation1', 'ToeSegmentation2', 'Trace', 'TwoLeadECG',
#     'TwoPatterns', 'UMD', 'UWaveGestureLibraryAll', 'uWaveGestureLibraryX', 'uWaveGestureLibraryY',
#     'uWaveGestureLibraryZ', 'wafer', 'Wine', 'WordSynonyms', 'Worms', 'WormsTwoClass', 'yoga'
# ]

# For testing, use subset
UNIVARIATE_DATASET_NAMES_2018 =  ['ArrowHead', 'Wine', 'FreezerSmallTrain', 'OliveOil',   'Car', 'NonInvasiveFetalECGThorax2', 
                                 'TwoPatterns', 'InsectWingbeatSound', 'BeetleFly', 'Yoga', 'InlineSkate', 'FaceAll',
                                 'EOGVerticalSignal', 'Ham', 'MoteStrain', 'ProximalPhalanxTW', 'WordSynonyms', 'Lightning7', 
                                 'GunPointOldVersusYoung', 'Earthquakes', 'FordB',]

ARCHIVE_NAMES = ['UCRArchive_2018']
DATASET_NAMES_FOR_ARCHIVE = {
    'UCRArchive_2018': UNIVARIATE_DATASET_NAMES_2018
}

# Model types
# CLASSIFIERS = ['teacher', 'student_kd', 'student_alone']
CLASSIFIERS = ['student_kd']  

ITERATIONS = {
    'teacher': 5,
    'student_kd': 5,
    'student_alone': 5
}

# Training hyperparameters
EPOCHS = 1500
BATCH_SIZE = 64
LEARNING_RATE = 0.001
PATIENCE = 50
MIN_LR = 0.0001
LR_FACTOR = 0.5

# Model hyperparameters
TEACHER_DEPTH = 6
STUDENT_DEPTH = 4
NB_FILTERS=32

BOTTLENECK_SIZE = 32
KERNEL_SIZE = 40

# Knowledge Distillation hyperparameters
ALPHA_LIST = [0.1, 0.3, 0.5, 0.7, 0.9]  # Weight for student loss vs distillation loss
TEMPERATURE_LIST = [4, 10, 16, 32]  # Temperature for softening predictions

# Paths (modify these for your system)
PATH_DATA = '/home/jabdullayev/phd/datasets/UCRArchive_2018/'
PATH_OUT = '/home/jabdullayev/phd/projects/KD4TSC/Results_Inception/'

# Device configuration
import torch
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Best teacher configuration
BEST_TEACHER_ONLY = True
PATH_TEACHER = os.path.join(PATH_OUT, 'results', 'teacher')