from Method import get_Matrix_Label, get_Matrix_Label_2, write_result
from model import DCPPS_testing

import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def predict(kinase):
    inputfile = 'DATASET/test_%s###.fasta' % (kinase)
    outputfile = "/result_kinase_model_{:s}".format(kinase)

    m = 25
    n = 25

    modelname = "./Best-weights/kinase_model_{:s}".format(kinase)
    modelweight = "kinase_model_{:s}".format(kinase)

    training_set = 'DATASET/train_%s###.fasta' % (kinase)

    _, _, _, _, _, X_val1, X_val2, Y_train_val = get_Matrix_Label(training_set, m, n)

    X_test_1, X_test_2, Y, name, indexs, site_types = get_Matrix_Label_2(inputfile, m, n)

    results = []
    result_probes = []
    Y_tests = []
    aucs = []
    auprs = []
    fprs = []
    tprs = []
    precisions_plts = []
    recalls_plts = []

    for i in range(5):
        t = i + 1
        result, result_probe, Y_test, auc, aupr, fpr, tpr, precisions_plt, recalls_plt = DCPPS_testing(
            X_test_1, X_test_2, Y, modelweight, X_val1, X_val2, Y_train_val, t)
        results.append(result)
        result_probes.append(result_probe)
        Y_tests.append(Y_test)
        aucs.append(auc)
        auprs.append(aupr)
        fprs.append(fpr)
        tprs.append(tpr)
        precisions_plts.append(precisions_plt)
        recalls_plts.append(recalls_plt)

    auc_ = max(aucs)
    index = aucs.index(auc_)
    result_ = results[index]
    result_probe_ = result_probes[index]
    Y_test_ = Y_tests[index]
    aupr_ = auprs[index]
    fpr_ = fprs[index]
    tpr_ = tprs[index]
    precisions_plt_ = precisions_plts[index]
    recalls_plt_ = recalls_plts[index]
    print(
        'The result of %s is:\nauc of roc:%.4f\nauc of pr:%.4f' % (modelname, auc_, aupr_))
    write_result(modelname, result_, result_probe_, Y_test_, fpr_, tpr_, precisions_plt_, recalls_plt_, auc_, aupr_,
                 outputfile)

    return auc_, aupr_


if __name__ == "__main__":

    ki = "MAPK"
    print("Now is prediction of", ki)
    predict(ki)
