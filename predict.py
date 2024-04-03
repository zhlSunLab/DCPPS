import sys
import pandas as pd
import numpy as np
import argparse
import csv
import numpy as np
from Method import get_Matrix_Label, get_Matrix_Label_2,write_result
from sklearn.metrics import roc_curve, auc
from model_DeepNet import DeepNet_testing

def predict(kinase):

    inputfile = 'DATASET/test_%s###.fasta' % (kinase)
    outputfile = "/result_general_model_{:s}".format('ST')

    m = 25
    n = 25


    modelname = "kinase_model_{:s}".format(kinase)
    modelweight = "kinase_model_{:s}".format(kinase)

    training_set = 'DATASET/train_%s###.fasta' % (kinase)

    _, _, _, _, _, X_val1, X_val2, Y_train_val = get_Matrix_Label(training_set, m, n)

    X_test_1, X_test_2, Y, name, indexs, site_types = get_Matrix_Label_2(inputfile, m, n)

    aucs = []
    auprs = []
    fprs = []
    tprs = []
    pres = []
    recalls = []
    F1s = []
    accs = []
    Sps = []
    Mccs = []

    for i in range(5):
        t = i + 1
        result, result_probe, Y_test, auc, aupr, fpr, tpr, precision, recall, F1, acc, Sp, mcc = DeepNet_testing(
            X_test_1, X_test_2, Y, modelweight, X_val1, X_val2, Y_train_val, t)
        aucs.append(auc)
        auprs.append(aupr)
        fprs.append(fpr)
        tprs.append(tpr)
        pres.append(precision)
        recalls.append(recall)
        F1s.append(F1)
        accs.append(acc)
        Sps.append(Sp)
        Mccs.append(mcc)

    auc_ = max(aucs)
    index = aucs.index(auc_)
    aupr_ = auprs[index]
    fpr_ = fprs[index]
    tpr_ = tprs[index]
    precision_ = pres[index]
    recall_ = recalls[index]
    F1 = F1s[index]
    acc = accs[index]
    Sp = Sps[index]
    mcc = Mccs[index]
    print(
        'The result of %s is:\nauc of roc:%.4f\nauc of pr:%.4f\nPrecision is %.4f\nF1 is %.4f\nAcc is %.4f\nSp is %.4f\nRecall is %.4f\nMcc is %.4f' % (
        modelname, auc_, aupr_, precision_, F1, acc, Sp, recall_, mcc))
    # write_result(modelname,fpr_,tpr_,precision_, recall_,outputfile)

    return auc_, aupr_, precision_, F1, acc, Sp, recall_, mcc

if __name__ == "__main__":

    predict('CAMK')