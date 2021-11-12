import numpy as np 
import struct 
import cmath as cm
import matplotlib.pyplot as plt 
import math
import re
from read_binary import read_binaryshort 
from light_progress.commandline import ProgressBar

#256サンプリング点ごとに離散フーリエ変換を行いパワースペクトルのリストを返す
def dft(data,N):
    res = []
    i=0
    
    print('dft progressing...')
    with ProgressBar(int(len(data)/N)) as progress_bar:
        while True:
            if (i+1)*N > len(data):
                break
            temp = data[i*N:(i+1)*N]
            x = []
            for k in range(N):
                w = cm.exp(-1j * 2 * cm.pi * k / N)
                X_k = 0
                for n in range(N):
                    X_k += temp[n] * (w ** n)
                x.append(math.sqrt(X_k.real**2+X_k.imag**2))
            res.append(x[:int(N/2)])
            i+=1
            progress_bar.forward()
    return np.array(res)

def calc_error(x,y):
    sum = 0
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            sum += (x[i][j]-y[i][j])**2
    return sum


def detect_h(x,w,iter_num):
    scale = ['C3','C#3','D3','D#3','E3','F3','F#3','G3','G#3','A3','A#3','B3','C4','C#4','D4','D#4','E4','F4']
    h = np.random.rand(w.shape[1],x.shape[1])*0.1
    log = []
    print(h.shape)
    wtx = np.dot(w.T,x)
    wtw = np.dot(w.T,w)
    with ProgressBar(iter_num) as progress_bar:
        for _ in range(iter_num):
            log.append(calc_error(x,np.dot(w,h)))
            wtwh = np.dot(wtw,h)
            for a in range(h.shape[0]):
                for u in range(h.shape[1]):
                    if wtwh[a][u]==0:
                        h[a][u]=0
                    else:
                        h[a][u] *= wtx[a][u]/wtwh[a][u]
            # print(h)
            progress_bar.forward()

    plt.plot(log)
    plt.xlabel('Number of iterations')
    plt.ylabel('error')
    plt.show()
    plt.plot(np.linspace(50,int(len(log)),len(log[50:])),log[50:])

    plt.xlabel('Number of iterations')
    plt.ylabel('error')
    plt.show()

    #音の数
    scale_num = 18
    for i in range(scale_num):
        scale_weight = np.zeros(h.shape[1])
        if i >= 10:
            marker = '+'
        else:
            marker = 'o'
        for j in range(int(h.shape[0]/scale_num)):
            scale_weight += h[i*int(h.shape[0]/scale_num)+j]
        for j in range(len(scale_weight)):
            if scale_weight[j]<=0.1:
                scale_weight[j]=0
        plt.plot(scale_weight,label=scale[i],marker = marker,markersize =4)
    plt.xlabel('Frame ID')
    plt.ylabel('H')
    plt.legend()
    plt.show()






def dft_all_data(filename):
    #一度に読み込むサンプリング点数
    size = 256
    data = read_binaryshort(filename)
    w_matrix = dft(data, size)
    w_matrix = w_matrix.T
    print(filename+' shape :',w_matrix.shape)
    return w_matrix

    
if __name__ == '__main__':
    w = dft_all_data('data/all_training_sounds.raw')
    # x = dft_all_data('data/em_chord.raw')

    x = dft_all_data('data/test.raw')
    detect_h(x, w,100)
