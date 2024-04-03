import numpy as np
import random

from sklearn.metrics import confusion_matrix
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
import tensorflow as tf
# tf.compat.v1.disable_eager_execution()
from tensorflow.python.keras.layers import Embedding
import pandas as pd
from sklearn.metrics import confusion_matrix

from sklearn import metrics

def file2str(filename):
    fr = open(filename)  # 打开文件
    numline = fr.readlines()  # 读取文件的行向量

    index = -1
    A = []
    F = []
    for eachline in numline:
        index += 1
        if '>' in eachline:
            A.append(index)
    A.append(index+1)

    B = []
    for eachline in numline:
        line = eachline.strip()
        listfoemline = line.split()
        B.append(listfoemline)

    name = []
    for i in range(len(A) - 1):
        K = A[i]
        input_sequence = str(B[K])
        input_sequence = input_sequence[3:-2]
        name.append(input_sequence)

    for i in range(len(A)-1):
        K = A[i]
        input_sequence = B[K + 1]
        input_sequence = str(input_sequence)
        input_sequence = input_sequence[1:-1]

        for j in range(A[i + 1] - A[i]):
            if K < A[i + 1] - 2:
                C = str(B[K + 2])
                input_sequence = input_sequence + C[1:-1]
                K += 1
        input_sequence = input_sequence.replace('\'', '')
        F.append(input_sequence)

    return name,F


def separt_positive(sequence, m, n):
    indexs = []
    for k in range(len(sequence)):
        sequence[k] = '****************************************************************************************************************************************************************************************************************************' + sequence[k] + '**********************************************************************************************************************************************************************************************************************************************************************************************'
        index = []
        kk = 0
        for i in range(len(sequence[k])):
            if sequence[k][i] == '#':
                index.append(i - kk - 1)
                kk += 1
        indexs.append(index)
    sub_sequences = []
    globalsa = []
    for i in range(len(sequence)):
        sequence[i] = sequence[i].translate(str.maketrans('', '', '#'))
        for k in range(len(indexs[i])):
            sub_sequence = sequence[i][indexs[i][k] - m:indexs[i][k] + n + 1]
            sub_sequences.append(sub_sequence)
            globalsa.append(i)
    return sub_sequences,np.array(globalsa)


def separt_negative(sequence, m, n):

    sequences = []
    for i in range(len(sequence)):
        if '#' in sequence[i]:
            sequences.append(sequence[i])
    sequence = sequences

    indexs = []
    for k in range(len(sequence)):
        sequence[k] = '**************************************************************************************************************************************************************************************************************************************************' + sequence[k] + '**************************************************************************************************************************************************************************************************************************************************'
        index = []
        kk = 0
        for i in range(len(sequence[k])):
            if sequence[k][i] == '#':
                index.append(i - kk - 1)
                kk += 1
        indexs.append(index)

    indexs2 = []
    for k in range(len(sequence)):

        sequence[k] = sequence[k].translate(str.maketrans('', '', '#'))
        index = []
        for i in range(len(sequence[k])):
            if sequence[k][i] == 'S' or sequence[k][i] == 'T':
                index.append(i)
        indexs2.append(index)
    indexs3 = []
    for i in range(len(indexs)):
        c = [x for x in indexs[i] if x in indexs2[i]]
        d = [y for y in (indexs[i] + indexs2[i]) if y not in c]
        indexs3.append(d)
    sub_sequences = []
    globals = []
    for i in range(len(sequence)):
        for k in range(len(indexs3[i])):
            sub_sequence = sequence[i][indexs3[i][k] - m:indexs3[i][k] + n + 1]
            sub_sequences.append(sub_sequence)
            globals.append(i)
    return sub_sequences,np.array(globals)


def separt_positive_2(sequence, m, n):
    indexs = []
    for k in range(len(sequence)):
        sequence[k] = '****************************************************************************************************************************************************************************************************************************' + sequence[k] + '**********************************************************************************************************************************************************************************************************************************************************************************************'
        index = []
        kk = 0
        for i in range(len(sequence[k])):
            if sequence[k][i] == '#':
                index.append(i - kk - 1)
                kk += 1
        indexs.append(index)
    sub_sequences = []
    globalsa = []
    for i in range(len(sequence)):
        sequence[i] = sequence[i].translate(str.maketrans('', '', '#'))
        for k in range(len(indexs[i])):
            sub_sequence = sequence[i][indexs[i][k] - m:indexs[i][k] + n + 1]
            sub_sequences.append(sub_sequence)
            globalsa.append(i)
    return sub_sequences,np.array(globalsa)


def separt_negative_2(sequence, m, n):

    sequences = []
    for i in range(len(sequence)):
        if '#' in sequence[i]:
            sequences.append(sequence[i])
        else:
            print('??')
    sequence = sequences

    indexs = []
    for k in range(len(sequence)):
        sequence[k] = '**************************************************************************************************************************************************************************************************************************************************' + sequence[k] + '**************************************************************************************************************************************************************************************************************************************************'
        index = []
        kk = 0
        for i in range(len(sequence[k])):
            if sequence[k][i] == '#':
                index.append(i - kk - 1)
                kk += 1
        indexs.append(index)

    indexs2 = []
    for k in range(len(sequence)):

        sequence[k] = sequence[k].translate(str.maketrans('', '', '#'))
        index = []
        for i in range(len(sequence[k])):
            if sequence[k][i] == 'Y':
                index.append(i)
        indexs2.append(index)
    indexs3 = []
    for i in range(len(indexs)):
        c = [x for x in indexs[i] if x in indexs2[i]]
        d = [y for y in (indexs[i] + indexs2[i]) if y not in c]
        indexs3.append(d)
    sub_sequences = []
    globals = []
    for i in range(len(sequence)):
        for k in range(len(indexs3[i])):
            sub_sequence = sequence[i][indexs3[i][k] - m:indexs3[i][k] + n + 1]
            sub_sequences.append(sub_sequence)
            globals.append(i)
    return sub_sequences,np.array(globals)


def get_test_file_ST(sequence, m, n):

    id = []
    indexs2 = []
    site_types = []

    indexs = []
    for k in range(len(sequence)):
        sequence[k] = '**************************************************************************************************************************************************************************************************************************************************' + sequence[k] + '**************************************************************************************************************************************************************************************************************************************************'
        index = []
        kk = 0
        for i in range(len(sequence[k])):
            if sequence[k][i] == '#':
                index.append(i - kk - 1)
                kk += 1
        indexs.append(index)


    for k in range(len(sequence)):
        sequence[k] = sequence[k].translate(str.maketrans('', '', '#'))
        index = []
        site_type = []
        for i in range(len(sequence[k])):
            if sequence[k][i] == 'S' or sequence[k][i] == 'T':
                index.append(i)
                site_type.append(sequence[k][i])
        indexs2.append(index)
        id.append(np.array(index) - 241)
        site_types.append(site_type)
    sub_sequences = []
    globals = []

    for i in range(len(sequence)):
        for k in range(len(indexs2[i])):
            sub_sequence = sequence[i][indexs2[i][k] - m:indexs2[i][k] + n + 1]
            sub_sequences.append(sub_sequence)
            globals.append(i)

    label = []
    for i in range(len(indexs)):
        for k in range(len(indexs2[i])):
            if indexs2[i][k] in indexs[i]:
                c = [0]
            else:c = [1]
            label.append(c)
    label = to_categorical(label)

    return sub_sequences, np.array(globals), label,id, site_types

def get_test_file_Y(sequence, m, n):
    id = []
    site_types = []
    indexs2 = []
    for k in range(len(sequence)):
        sequence[
            k] = '**************************************************************************************************************************************************************************************************************************************************' + \
                 sequence[
                     k] + '**************************************************************************************************************************************************************************************************************************************************'

        index = []
        site_type = []
        for i in range(len(sequence[k])):
            if sequence[k][i] == 'Y':
                index.append(i)
                site_type.append(sequence[k][i])
        indexs2.append(index)
        site_types.append(site_type)
        id.append(np.array(index) - 241)
    sub_sequences = []
    globals = []
    for i in range(len(sequence)):
        for k in range(len(indexs2[i])):
            sub_sequence = sequence[i][indexs2[i][k] - m:indexs2[i][k] + n + 1]
            sub_sequences.append(sub_sequence)
            globals.append(i)

    return sub_sequences,np.array(globals), id, site_types

def str2dic(input_sequence):
    char = sorted(
        ['G', 'A', 'V', 'L', 'I', 'P', 'F', 'Y', 'W', 'S', 'T', 'C', 'M', 'N', 'Q', 'D', 'E', 'K', 'R', 'H'])

    char_to_index = {}
    index = 1
    result_index = []
    for c in char:
        char_to_index[c] = index
        index = index + 1
    char.append('*')
    char.append('U')
    char.append('B')
    char_to_index['*'] = 0
    char_to_index['U'] = char_to_index['D']
    char_to_index['B'] = char_to_index['D']
    for word in input_sequence:
        result_index.append(char_to_index[word])
    return result_index


def vec_to_onehot(mat,pc,kk,mmm=51):
    m = len(mat)
    return_mat = np.zeros((m, mmm, kk))
    for i in range(len(mat)):
        metrix = np.zeros((mmm, kk))
        for j in range(len(mat[i])):
            metrix[j] = pc[mat[i][j]]
        return_mat[i,:,:] = metrix
    return return_mat

def get_Matrix_Label (filename, m, n):


    name, input_sequence = file2str(filename)
    input_sequence_2 = np.copy(input_sequence)
    input_sequence_3 = np.copy(input_sequence)

    sequence_positive, globals_positive = separt_positive(input_sequence, m, n)
    sequence_negative, globals_negative = separt_negative(input_sequence_2, m, n)

    num_positive = len(sequence_positive)
    num_negative = len(sequence_negative)

    num_val = int(num_positive/10)

    X_train_positive = []
    for i in range(len(sequence_positive)):
        result_index = str2dic(sequence_positive[i])
        X_train_positive.append(result_index)

    X_train_negative = []
    for i in range(len(sequence_negative)):
        result_index = str2dic(sequence_negative[i])
        X_train_negative.append(result_index)

    random.seed(0)
    X_train_positive = np.array(X_train_positive)
    X_train_negative = np.array(X_train_negative)

    ls = list((range(len(X_train_positive))))
    random.shuffle(ls)

    X_val_positive = X_train_positive[ls][:num_val]
    X_train_positive = X_train_positive[ls][num_val:]

    global_val_positive = globals_positive[ls][:num_val]
    global_train_positive = globals_positive[ls][num_val:]

    random.seed(1)
    ls2 = list((range(len(X_train_negative))))
    random.shuffle(ls2)
    X_val_negative = X_train_negative[ls2][:num_val]
    X_train_negative = X_train_negative[ls2][num_val:]

    global_val_negative = globals_negative[ls2][:num_val]
    global_train_negative = globals_negative[ls2][num_val:]
    X_val = np.vstack((X_val_positive, X_val_negative))
    for kk in range(len(input_sequence_3)):
        input_sequence[kk] = input_sequence_3[kk].translate(str.maketrans('', '', '#'))

    globel_input_sequence = []
    for kk in range(len(input_sequence)):
        result_index = str2dic(input_sequence[kk])
        globel_input_sequence.append(result_index)

    input_sequence = pad_sequences(globel_input_sequence,maxlen = 2000)
    X_val2 = np.vstack((input_sequence[global_val_positive],input_sequence[global_val_negative]))
    # blosum_ = data_blosum()
    # X_train_positive = blosum_.trans(X_train_positive)
    # X_train_negative = blosum_.trans(X_train_negative)
    # global_train_positive = blosum_.trans(global_train_positive)
    # global_train_negative = blosum_.trans(global_train_negative)

    Y = [0]*(num_positive-num_val)+[1]*(num_negative-num_val)
    Y_val = [0]*num_val+[1]*num_val

    Y_val = to_categorical(Y_val)
    return X_train_positive, X_train_negative, global_train_positive, global_train_negative, Y, X_val, X_val2,Y_val


def get_Matrix_Label_2 (filename, m, n):

    name, input_sequence = file2str(filename)

    sequence, globals,label, indexs, site_types = get_test_file_ST(input_sequence, m, n)

    X_train = []
    for i in range(len(sequence)):
        result_index = str2dic(sequence[i])
        X_train.append(result_index)

    random.seed(0)
    X_test = np.array(X_train)

    name, input_sequence = file2str(filename)

    globel_input_sequence = []

    for kk in range(len(input_sequence)):
        input_sequence[kk] = input_sequence[kk].translate(str.maketrans('', '', '#'))
        result_index = str2dic(input_sequence[kk])
        globel_input_sequence.append(result_index)
    input_sequence = pad_sequences(globel_input_sequence, maxlen=2000)
    X_test2 = input_sequence[globals]


    return X_test, X_test2,label,name, indexs, site_types


def get_label(pos,neg):
    num_positive = len(pos)
    num_negative = len(neg)
    Y = [0]*(num_positive)+[1]*(num_negative)
    Y = to_categorical(Y)
    return Y


def encode_protein_sequence(sequence):
    encoding = {'A': '01000000', 'R': '01100000', 'N': '01110000', 'D': '10000000',
                'C': '10010000', 'Q': '10100000', 'E': '10110000', 'G': '11000000',
                'H': '11010000', 'I': '11100000', 'L': '11110000', 'K': '01010000',
                'M': '01001000', 'F': '01000100', 'P': '01000010', 'S': '01000001',
                'T': '01100001', 'W': '01100010', 'Y': '01100100', 'V': '01101000'}

    encoded_sequence = ''
    for amino_acid in sequence:
        encoded_sequence += encoding.get(amino_acid, '00000000')

    return encoded_sequence


def encode_dna_sequence(sequence):
    encoding = {'A': '00', 'G': '01', 'T': '10', 'C': '11'}

    encoded_sequence = ''
    for base in sequence:
        encoded_sequence += encoding.get(base, '00')

    return encoded_sequence
# def load_pretrained_embs(embfile):
#     # with open(embfile, encoding='utf-8') as f:
#     #         lines = f.readlines()
#     #         items = lines[0].split()
#
#     B = 'ACDEFGHIKLMNPQRSTVWYO'
#     B = list(B)
#     index = len(B)
#     reverse = lambda x: dict(zip(x, range(len(x))))
#     id = reverse(B)
#     embeddings = np.zeros((len(B), 21))
#
#     for line in B:
#             embeddings[id[line]] = blosum_mat[line]
#     return embeddings

# blosum_mat=pd.read_csv("./blosum.csv",header=0,index_col=0)
# word2vec_path = "./blosum.csv"

# class data_blosum:
#     def __init__(self):
#         self.blosum = Embedding(21, 21)
#         weight = load_pretrained_embs(word2vec_path)
#         self.blosum = self.blosum.set_weights(weight)
#
#     def trans(self,word_ids):
#
#         x = self.blosum(word_ids)
#         return x

def metric_10(y_true,y_pred):
 TN=tf.reduce_sum(y_true*y_pred)
 TP=tf.reduce_sum((1-y_true)*(1-y_pred))
 FN=tf.reduce_sum((1-y_true)*y_pred)
 FP=tf.reduce_sum(y_true*(1-y_pred))
 return TP,TN,FP,FN

def write_result(modelname,fpr,tpr,precision, recall,outputfile):
    with open(modelname + outputfile, 'w') as f:
        num = 0
        for u in range(len(precision)):
            if u < len(fpr)-1 :
                f.write(str(precision[num]))
                f.write('\t')
                f.write(str(recall[num]))
                f.write('\t')
                f.write(str(fpr[num]))
                f.write('\t')
                f.write(str(tpr[num]))
                f.write('\n')
            else:
                f.write(str(precision[num]))
                f.write('\t')
                f.write(str(recall[num]))
                f.write('\n')
            num = num + 1

    print('Successfully predict the phosphorylation site ! prediction results are stored in ' + modelname + outputfile)


def positional_embedding(inputs,zero_pad=True,scaled=True):
    """

    :param inputs: after embedding->(batch_size,seq_len,num_units)
    :param scaled:a bool value,whther to /np.sqrt(num_units)
    :return:(batch_size,seq_len,features)
    """
    # N, T ,num_units = inputs.get_shape().as_list()
    N = tf.shape(inputs)[0]
    batch_size, T ,num_units = inputs.get_shape().as_list()
    # num_units = hp.hidden_units
    with tf.compat.v1.variable_scope('positinal_embedding',reuse=tf.compat.v1.AUTO_REUSE):
        position_ind = tf.tile(tf.expand_dims(tf.range(T), 0), [N,1])#generate a sequence tuple like (0,1,2,3...)->(N,T)
        # Sinusoidal Positional_Encoding
        # PE = np.array([[pos/np.power(10000,2.*i/num_units) for i in range(num_units)] for pos in range(T)])
        PE = np.array([
            [pos / np.power(10000, (i-i%2)/num_units) for i in range(num_units)]
            for pos in range(T)])
        PE[:, 0::2] = np.sin(PE[:, 0::2])
        PE[:, 1::2] = np.cos(PE[:, 1::2])
        lookup_table = tf.convert_to_tensor(PE,dtype=tf.float32)#(seq_len,num_units),no train
        if zero_pad:
            lookup_table = tf.concat((tf.zeros(shape=[1, num_units]),
                                      lookup_table[1:, :]), 0)
        outputs = tf.nn.embedding_lookup(lookup_table, position_ind)
        if scaled:
            outputs = outputs * num_units ** 0.5
        outputs = tf.cast(outputs,'float32')#positinal embedding may be to float64
    return outputs