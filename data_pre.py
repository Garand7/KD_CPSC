# import h5py
import numpy as np
import time
import scipy.io as scio
import librosa
from matplotlib import pyplot as plt

slice_len = 1024
slice_step = slice_len
FFT_num = 64
hop_length = 32

path = 'C://Users//ROG//Desktop//CPSC//TrainingSet'
endNum = 11

s_total = 0
v_total = 0
x_total = 0

start = time.time()

XTrain = []
YTrain = []
YTrain_Bilstm = []
XTrain_abnormal_lstm = []
YTrain_abnormal_lstm = []
for namei in range(1, endNum):
    print(namei)
    f1 = path+'/data/A{:0>2d}.mat'.format(namei)
    f2 = path+'/ref/R{:0>2d}.mat'.format(namei)

    X = scio.loadmat(f1)['ecg']
    Y = scio.loadmat(f2)['ref']
    Y_S = Y[0, 0]['S_ref']
    Y_V = Y[0, 0]['V_ref']

    len_slice = len(X) // slice_step - (slice_len//slice_step) + 1
    XTrain_temp = []

    for slice in range(len_slice):
        print(slice)
        temp = X[slice*slice_step:slice*slice_step+slice_len].squeeze()
        x = librosa.core.stft(temp, n_fft=FFT_num, hop_length=hop_length, win_length=FFT_num, window='hamming',
                              center=False)
        # x = x[0:x.shape[0]//2]
        # plt.contourf(abs(x))
        # plt.show()
        XTrain_temp.append(np.array(abs(x)).transpose())

    print('XTrain end')
    YTrain_temp = np.zeros((len_slice,))

    len_lstm = slice_len//hop_length - (FFT_num//hop_length - 1)
    YTrain_lstm_temp = np.zeros((len_slice, len_lstm, 1))

    m = 0
    n = 0
    p = len(Y_S) - 1
    q = len(Y_V) - 1
    Y_S = Y_S - 1
    Y_V = Y_V - 1

    def trunum(index_cal):
        if index_cal >= len_lstm:
            index_cal = len_lstm - 1
        return index_cal


    if p >= 0:
        while Y_S[m] < slice_step:
            YTrain_temp[0] = 1
            YTrain_lstm_temp[0, trunum(int((Y_S[m])//hop_length)), 0] = 2
            m += 1

        while Y_S[p] >= len_slice * slice_step:
            YTrain_temp[len_slice-1] = 1
            YTrain_lstm_temp[0, trunum(int((Y_S[p]-(len_slice-1) * slice_step)//hop_length)), 0] = 2
            p -= 1
        if slice_step == slice_len // 2:
            for si in range(m, p+1):
                y_position = Y_S[si] // slice_step
                YTrain_temp[y_position] = 1
                YTrain_temp[y_position - 1] = 1
                YTrain_lstm_temp[y_position, trunum(int((Y_S[si]-y_position * slice_step) // hop_length)), 0] = 2
                YTrain_lstm_temp[y_position - 1, trunum(int((Y_S[si]-(y_position-1) * slice_step) // hop_length)), 0] = 2
        else:
            for si in range(m, p + 1):
                y_position = Y_S[si] // slice_step
                YTrain_temp[y_position] = 1
                YTrain_lstm_temp[y_position, trunum(int((Y_S[si] - y_position * slice_step) // hop_length)), 0] = 2

    if q >= 0:
        while Y_V[n] < slice_step:
            YTrain_temp[0] = 1
            YTrain_lstm_temp[0, trunum(int((Y_V[m])//hop_length)), 0] = 1
            n += 1
        while Y_V[q] >= len_slice * slice_step:
            YTrain_temp[len_slice-1] = 1
            YTrain_lstm_temp[0, trunum(int((Y_V[q]-(len_slice-1) * slice_step)//hop_length)), 0] = 1
            q -= 1
        if slice_step == slice_len // 2:
            for vi in range(n, q+1):
                y_position = Y_V[vi] // slice_step
                YTrain_temp[y_position] = 1
                YTrain_temp[y_position - 1] = 1
                YTrain_lstm_temp[y_position, trunum(int((Y_V[vi]-y_position * slice_step) // hop_length)), 0] = 1
                YTrain_lstm_temp[y_position - 1, trunum(int((Y_V[vi]-(y_position-1) * slice_step) // hop_length)), 0] = 1
        else:
            for vi in range(n, q + 1):
                y_position = Y_V[vi] // slice_step
                YTrain_temp[y_position] = 1
                YTrain_lstm_temp[y_position, trunum(int((Y_V[vi] - y_position * slice_step) // hop_length)), 0] = 1

    print('YTrain end')
    XTrain.extend(XTrain_temp)
    YTrain.extend(YTrain_temp)
    YTrain_Bilstm.extend(YTrain_lstm_temp)

    XTrain_abnormal_lstm_temp = []
    YTrain_abnormal_lstm_temp = []
    for labeli in range(len_slice):
        if YTrain_temp[labeli] != 0:
            XTrain_abnormal_lstm_temp.append(XTrain_temp[labeli])
            YTrain_abnormal_lstm_temp.append(YTrain_lstm_temp[labeli])
    print('lstm end')
    XTrain_abnormal_lstm.extend(XTrain_abnormal_lstm_temp)
    YTrain_abnormal_lstm.extend(YTrain_abnormal_lstm_temp)

    elapsed = time.time() - start
    print("Time used:", elapsed)

XTrain = np.array(XTrain)
YTrain = np.array(YTrain)
YTrain_Bilstm = np.array(YTrain_Bilstm)
XTrain_abnormal_lstm = np.array(XTrain_abnormal_lstm)
YTrain_abnormal_lstm = np.array(YTrain_abnormal_lstm)

f3 = 'XTrain'
f4 = 'YTrain'
f5 = 'YTrain_Bilstm'
f6 = 'XTrain_abnormal_lstm'
f7 = 'YTrain_abnormal_lstm'

np.save(f3, XTrain)
np.save(f4, YTrain)
np.save(f5, YTrain_Bilstm)
np.save(f6, XTrain_abnormal_lstm)
np.save(f7, YTrain_abnormal_lstm)
