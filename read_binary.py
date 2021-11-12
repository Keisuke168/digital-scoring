import struct
import numpy as np
#バイナリファイル読み込み用関数
def read_binaryfloat(filename):
    f=open(filename, 'rb')
    x = []
    while True:
        #4yteごとfloatで読み込む
        temp_x = f.read(4)
        if not temp_x:
            break
        x.append(struct.unpack('f',temp_x)[0])
    return np.array(x)

def read_binaryshort(filename):
    f=open(filename, 'rb')
    data=[]
    while True:
        #2byteごと読み込む
        temp = f.read(2)
        if not temp:
            break
        data.append(struct.unpack('h',temp)[0])
    return np.array(data)