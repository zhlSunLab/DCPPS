import matplotlib
matplotlib.use('Agg')
from Method import get_Matrix_Label
from model_DeepNet import DeepNet_training
import argparse

# lr = 0.005,dropout = 0.5

def train(kinase):
    #
    # parser = argparse.ArgumentParser(
    #     description='DeepPSP: a prediction tool for general, kinase-specific phosphorylation prediction')
    # parser.add_argument('-dataset', dest='kinase', type=str,
    #                     help='if -train-type is \'kinase\', -kinase indicates the specific kinase.',
    #                     required=False, default=None)
    #
    # args = parser.parse_args()
    # kinase = args.kinase
    inputfile = 'DATASET/train_{:s}###.fasta'.format(kinase)

    m = 25
    n = 25

    X_train_positive, X_train_negative, global_train_positive, global_train_negative, Y, X_val1, X_val2,Y_val = get_Matrix_Label(inputfile, m, n)

    modelname = "kinase_model_{:s}".format(kinase)
    for i in range(1):
        t = i + 1
        DeepNet_training(inputfile, X_train_positive, X_train_negative, global_train_positive,
                                global_train_negative, Y, X_val1, X_val2, Y_val, modelname,t)


if __name__ == "__main__":

    train('CAMK')