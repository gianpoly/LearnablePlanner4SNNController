import sys
sys.path.append('./IO')
from amc_processing import construct_CMU_train_set

recording_file="./data/walk.amc"
construct_CMU_train_set(recording_file, './data/training_sequence.txt')
recording = './data/training_sequence.txt'