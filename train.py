import matplotlib

matplotlib.use('Agg')
from Method import get_Matrix_Label
from model import DCPPS_training

import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def train(kinase):
    inputfile = 'DATASET/train_{:s}###.fasta'.format(kinase)

    m = 25
    n = 25

    X_train_positive, X_train_negative, global_train_positive, global_train_negative, Y, X_val1, X_val2, Y_val = get_Matrix_Label(
        inputfile, m, n)

    modelname = "kinase_model_{:s}".format(kinase)
    for i in range(5):
        t = i + 1
        DCPPS_training(inputfile, X_train_positive, X_train_negative, global_train_positive,
                       global_train_negative, Y, X_val1, X_val2, Y_val, modelname, t)


if __name__ == "__main__":

    train('MAPK')
