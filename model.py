from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from tensorflow.keras.layers import *

from Method import *
import os
from sklearn.linear_model import LogisticRegression
import numpy as np
from tensorflow.keras import backend as K
import tensorflow as tf

dropout = 0.5
learn_rate = 0.005


def expand_dim_backend(x):
    x1 = K.reshape(x, (-1, 1, 100))
    return x1


def expand_dim_backend2(x):
    x1 = K.reshape(x, (-1, 1, 51))
    return x1


def expand_dim_backend3(x):
    x1 = K.reshape(x, (-1, 1, 200))
    return x1


def multiply(a):
    x = tf.multiply(a[0], a[1])
    return x


def res_block(input, stride, filter):
    residual = input
    residual = Conv1D(strides=stride, kernel_size=1, filters=filter, padding='same')(residual)
    residual = BatchNormalization()(residual)
    residual = Activation('relu')(residual)
    residual = Dropout(dropout)(residual)
    return residual


def FFN(x, dense):
    l_x = LayerNormalization(axis=-1)(x)
    d_x = Dense(dense, activation='swish')(l_x)
    d_x = Dropout(dropout)(d_x)
    d_x = Dense(dense, activation='swish')(d_x)
    d_x = Dropout(dropout)(d_x)
    return d_x


def CRIU(x1, x2, dense):
    if dense > 51:
        dense1 = 51
    else:
        dense1 = 200
    x1_ = FFN(x1, dense)
    x1_ = Add()([x1_, res_block(x1, 1, dense)])
    x1_ = Activation('relu')(x1_)
    x2_ = FFN(x2, dense1)
    x2_ = Add()([x2_, res_block(x2, 1, dense1)])
    x2_ = Activation('relu')(x2_)

    y1 = BatchNormalization(axis=-1)(x1_)
    y2 = BatchNormalization(axis=-1)(x2_)
    y = MultiHeadAttention(2, 100, 2)(y1, y2, y2)
    y = Dropout(dropout)(y)

    y = Add()([y, res_block(y1, 1, dense)])
    y = Activation('relu')(y)

    X_conv = Conv1D(strides=1, kernel_size=1, filters=100, padding='same')(y)
    X_conv = Add()([X_conv, res_block(y, 1, 100)])
    X_conv = BatchNormalization(axis=-1)(X_conv)  # 🍒
    X_conv = Activation('relu')(X_conv)

    X_dense = Dense(dense, activation='relu')(X_conv)
    X_dense = Dropout(dropout)(X_dense)

    x_ = FFN(X_dense, dense)
    x_ = Add()([x_, res_block(X_dense, 1, dense)])
    x_ = LayerNormalization(axis=-1)(x_)
    x_ = Activation('relu')(x_)
    return x_


def DEE(input_data):
    pos = Lambda(positional_embedding)(input_data)
    out_data = pos + input_data
    out_data = LayerNormalization()(out_data)
    return out_data


def create_model(input_shape_1=[51], input_shape_2=[2000], unit=128, filter=100):
    X_input_1 = Input(shape=input_shape_1)
    X_input_2 = Input(shape=input_shape_2)

    #########DEE#######
    X_input_1_em = Embedding(input_dim=51, output_dim=21, input_length=51)(X_input_1)
    X_input_2_em = Embedding(input_dim=2000, output_dim=21, input_length=2000)(X_input_2)
    X_input_1_em = DEE(X_input_1_em)
    X_input_2_em = DEE(X_input_2_em)

    ##########Stem#############
    X_conv_1 = Conv1D(strides=1, kernel_size=1, filters=filter, padding='same')(X_input_1_em)
    X_conv_1 = Add()([X_conv_1, res_block(X_input_1_em, 1, 100)])
    X_conv_1 = BatchNormalization(axis=-1)(X_conv_1)  # 🍒
    X_conv_1 = Activation('relu')(X_conv_1)
    X_conv_1 = Dropout(dropout)(X_conv_1)

    #############AERU-1###############
    squeeze_1 = GlobalAveragePooling1D()(X_conv_1)
    squeeze_1 = Lambda(expand_dim_backend)(squeeze_1)

    excitation_1 = Conv1D(filters=16, kernel_size=1, strides=1, padding='valid', activation='relu')(squeeze_1)
    excitation_1 = Conv1D(filters=100, kernel_size=1, strides=1, padding='valid', activation='sigmoid')(excitation_1)

    excitation_1 = Dropout(dropout)(excitation_1)

    X = Lambda(multiply)([X_conv_1, excitation_1])
    X1 = Permute([2, 1])(X)

    X_res_2 = res_block(X_conv_1, 1, 100)
    X_res_2 = Permute([2, 1])(X_res_2)
    X_res_2 = Add()([X1, X_res_2])
    X_res_2 = Activation('relu')(X_res_2)

    ############AERU-2############
    squeeze_2 = GlobalAveragePooling1D()(X_res_2)
    squeeze_2 = Lambda(expand_dim_backend2)(squeeze_2)

    excitation_2 = Conv1D(filters=16, kernel_size=1, strides=1, padding='valid', activation='relu')(squeeze_2)
    excitation_2 = Conv1D(filters=51, kernel_size=1, strides=1, padding='valid', activation='sigmoid')(excitation_2)

    excitation_2 = Dropout(dropout)(excitation_2)

    X2 = Lambda(multiply)([X_res_2, excitation_2])
    X2 = Permute([2, 1])(X2)

    X_res_3 = Permute([2, 1])(X_res_2)
    X_res_3 = res_block(X_res_3, 1, 100)
    X_res_3 = Add()([X2, X_res_3])
    X_res_3 = Activation('relu')(X_res_3)

    ###########FC-1###########
    X2_ = GlobalAveragePooling1D()(X_res_3)
    X_Dense_1 = Dense(64, activation='relu')(X2_)

    #############BiLSTM-1###########
    X_Bi_LSTM_1 = Bidirectional(LSTM(unit, return_sequences=True))(Permute([2, 1])(X_res_2))
    X_Bi_LSTM_1 = Dropout(dropout)(X_Bi_LSTM_1)
    X_Bi_LSTM_1 = BatchNormalization()(X_Bi_LSTM_1)

    X_conv_2 = Conv1D(strides=10, kernel_size=15, filters=filter, padding='same')(X_input_2_em)
    X_res_4 = res_block(X_input_2_em, 10, 100)
    X_res_4 = Add()([X_conv_2, X_res_4])
    X_conv_2 = BatchNormalization(axis=-1)(X_res_4)  # 🍒
    X_conv_2 = Activation('relu')(X_conv_2)
    X_conv_2 = Dropout(dropout)(X_conv_2)

    #########AERU-3############
    squeeze_3 = GlobalAveragePooling1D()(X_conv_2)
    squeeze_3 = Lambda(expand_dim_backend)(squeeze_3)

    excitation_3 = Conv1D(filters=16, kernel_size=1, strides=1, padding='valid', activation='relu')(squeeze_3)
    excitation_3 = Conv1D(filters=100, kernel_size=1, strides=1, padding='valid', activation='sigmoid')(excitation_3)

    excitation_3 = Dropout(dropout)(excitation_3)

    X3 = Lambda(multiply)([X_conv_2, excitation_3])
    X3 = Permute([2, 1])(X3)

    X_res_5 = res_block(X_conv_2, 1, 100)
    X_res_5 = Permute([2, 1])(X_res_5)
    X_res_5 = Add()([X3, X_res_5])
    X_res_5 = Activation('relu')(X_res_5)

    #############BiLSTM-2############
    X_Bi_LSTM_2 = Bidirectional(LSTM(unit, return_sequences=True))(Permute([2, 1])(X_res_5))
    X_Bi_LSTM_2 = Dropout(dropout)(X_Bi_LSTM_2)
    X_Bi_LSTM_2 = BatchNormalization()(X_Bi_LSTM_2)

    cross_attenion_1 = CRIU(Permute([2, 1])(X_Bi_LSTM_1), Permute([2, 1])(X_Bi_LSTM_2), 51)
    cross_attenion_2 = CRIU(Permute([2, 1])(X_Bi_LSTM_2), Permute([2, 1])(X_Bi_LSTM_1), 200)

    ##########RB-1############
    X_res_6 = res_block(Permute([2, 1])(X_Bi_LSTM_1), 1, 51)
    X_res_6 = Add()([cross_attenion_1, X_res_6])
    X_res_6 = Activation('relu')(X_res_6)
    X_res_6 = GlobalAveragePooling1D()(X_res_6)

    ##########RB-2############
    X_res_7 = res_block(Permute([2, 1])(X_Bi_LSTM_2), 1, 200)
    X_res_7 = Add()([cross_attenion_2, X_res_7])
    X_res_7 = Activation('relu')(X_res_7)
    X_res_7 = GlobalAveragePooling1D()(X_res_7)

    ###########FC-2,3###########
    X_Dense_2 = Dense(64, activation='relu')(X_res_6)
    X_Dense_3 = Dense(64, activation='relu')(X_res_7)

    ############AERU-4###########
    squeeze_4 = GlobalAveragePooling1D()(X_res_5)
    squeeze_4 = Lambda(expand_dim_backend3)(squeeze_4)

    excitation_4 = Conv1D(filters=16, kernel_size=1, strides=1, padding='valid', activation='relu')(squeeze_4)
    excitation_4 = Conv1D(filters=200, kernel_size=1, strides=1, padding='valid', activation='sigmoid')(excitation_4)

    excitation_4 = Dropout(dropout)(excitation_4)

    X4 = Lambda(multiply)([X_res_5, excitation_4])
    X4 = Permute([2, 1])(X4)

    X_res_8 = Permute([2, 1])(X_res_5)
    X_res_8 = res_block(X_res_8, 1, 100)
    X_res_8 = Add()([X4, X_res_8])
    X_res_8 = Activation('relu')(X_res_8)

    ###########FC-4###########
    X4_ = GlobalAveragePooling1D()(X_res_8)
    X_Dense_4 = Dense(64, activation='relu')(X4_)

    ###########concat###########
    XX = Concatenate()([X_Dense_1, X_Dense_2, X_Dense_3, X_Dense_4])

    ###########FFN###########
    out = Dense(32)(XX)
    out = Activation('relu')(out)
    out = Dense(2)(out)
    out = Activation('softmax')(out)

    model_3 = Model(inputs=[X_input_1, X_input_2], outputs=[out])

    return model_3


def DCPPS_training(training_set_name, X_train_positive, X_train_negative, global_train_positive,
                     global_train_negative, Y, X_val1, X_val2, Y_val, modelname, t):
    num = int(len(X_train_negative) / len(X_train_positive))
    print(len(X_train_negative))
    print(len(X_train_positive))

    if num > 25:
        num = 25

    for i in range(num):
        inputfile = (training_set_name)
        _, input_sequence = file2str(inputfile)

        for kk in range(len(input_sequence)):
            input_sequence[kk] = input_sequence[kk].translate(str.maketrans('', '', '#'))

        globel_input_sequence = []
        for kk in range(len(input_sequence)):
            result_index = str2dic(input_sequence[kk])
            globel_input_sequence.append(result_index)
        input_sequence = pad_sequences(globel_input_sequence, maxlen=2000)

        if len(X_train_positive) * (i + 1) < len(X_train_negative):
            X_train = np.vstack(
                (X_train_positive, X_train_negative[len(X_train_positive) * i:len(X_train_positive) * (i + 1)]))
            X_train2 = np.vstack((input_sequence[global_train_positive], input_sequence[
                global_train_negative[len(X_train_positive) * i:len(X_train_positive) * (i + 1)]]))

        else:

            X_train = np.vstack((X_train_positive, X_train_negative[len(X_train_negative) - len(X_train_positive):]))
            X_train2 = np.vstack((input_sequence[global_train_positive], input_sequence[
                global_train_negative[len(X_train_negative) - len(X_train_positive):]]))

        print('This is the %d-th iteration' % (i + 1))

        print('The number of training set is %d' % (len(X_train)), 'The number of validation set is %d' % (len(X_val1)))

        Y = [0] * len(X_train_positive) + [1] * len(X_train_positive)
        Y = to_categorical(Y)

        model = create_model([51], [2000], 128, 100)

        adam = Adam(lr=learn_rate, beta_1=0.9, beta_2=0.999, decay=0)
        model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

        filepath1 = 'model_weight' + '/' + modelname

        if not os.path.exists(filepath1):
            os.makedirs(filepath1)
        filepath2 = filepath1 + '/' + 'weights%d-seed%s.h5' % ((i + 1), t)

        checkpoint = ModelCheckpoint(filepath=filepath2, monitor='val_accuracy', verbose=1, save_best_only=True,
                                     mode='max')

        earlystopping = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=100, verbose=1, mode='auto')
        callbacks_list = [checkpoint, earlystopping]

        model.fit([X_train, X_train2], Y, validation_data=([X_val1, X_val2], Y_val),
                  epochs=1000, callbacks=callbacks_list, batch_size=128, verbose=2, shuffle=True)
        K.clear_session()


def DCPPS_testing(X_test, X_test_2, Y, modelname, X_val1, X_val2, Y_train_val, t):
    weight_file = "./Best-weights/" + modelname

    X_predict_test = np.zeros((len(X_test), 2 * len(os.listdir(weight_file))))
    X_predict_val = np.zeros((len(X_val1), 2 * len(os.listdir(weight_file))))

    Y_test = [np.argmax(one_hot) for one_hot in Y]
    Y_test = np.array(Y_test)
    Y_train = [np.argmax(one_hot) for one_hot in Y_train_val]
    Y_train = np.array(Y_train)

    N = len(os.listdir(weight_file))
    if N > 25:
        N = 25

    for i in range(N):
        print('this is %d th ' % (i + 1))

        model = create_model([51], [2000], 128, 100)
        model.load_weights(weight_file + '/' + 'weights%d-seed%s.h5' % ((i + 1), t))
        adam = Adam(lr=learn_rate, beta_1=0.9, beta_2=0.999, decay=0)
        model.compile(loss='sparse_categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

        X_predict_test[:, i * 2:(i + 1) * 2] = model.predict([X_test, X_test_2])
        X_predict_val[:, i * 2:(i + 1) * 2] = model.predict([X_val1, X_val2])
        print("now is %s:" % modelname)
        K.clear_session()

    lr = LogisticRegression(C=0.1)

    lr.fit(X_predict_val, Y_train)
    print("now is %s:" % modelname)

    predict_ = lr.predict(X_predict_test)
    predict_probe_ = lr.predict_proba(X_predict_test)

    print('***********************print final result*****************************')
    fpr, tpr, threshold = roc_curve(Y_test, predict_probe_[:, 0], pos_label=0)
    roc_auc = auc(fpr, tpr)
    precisions_plt, recalls_plt, thresholds = precision_recall_curve(Y_test, predict_probe_[:, 0], pos_label=0)
    auc_pr = auc(recalls_plt, precisions_plt)
    print(
        'The %dth test result of %s is:\nauc of roc:%.4f\nauc of pr:%.4f' % (t, modelname, roc_auc, auc_pr))

    return predict_, predict_probe_[:, 0], Y_test, roc_auc, auc_pr, fpr, tpr, precisions_plt, recalls_plt
