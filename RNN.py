# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 15:34:38 2018

@author: remoh
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.layers.recurrent import GRU
from keras.models import Sequential
from keras import backend as K
from keras.layers import Dense, Activation
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from datetime import datetime as dt
from glob import glob
from sklearn.decomposition import PCA

class body:
    def __init__(self,dir_name):
        #データロード
        # 関節名、そのデータフレームからなる辞書
        self.joint_dict = {}
        for joint_file in glob(f"{dir_name}/*.csv"):
            joint_name = joint_file.split('/')[-1].split('\\')[1].split('.')[0]
            print(f'loading {joint_name}...')
            print(f"{dir_name}/{joint_name}.csv")
            self.joint_dict[joint_name] = pd.read_csv(f"{dir_name}/{joint_name}.csv", comment='/')
        
        for dataframe in self.joint_dict.values():
            if isinstance(dataframe.time[0], str):
                times = np.array([dt.strptime(t, '%H:%M:%S.%f') for t in dataframe.time])
            else:
                times = np.array([dt.strptime(t, '%H%M%S%f') for t in dataframe.time.astype(str)])
            times = times - times[0]
            dataframe.time = [t.total_seconds() for t in times]
        
    def reldata(self, joint1, joint2="SpineBase"):
        return self.joint_dict[joint1].iloc[:,1:] - self.joint_dict[joint2].iloc[:,1:] 

#tRange区切りでpca(n_component=3)->headのみ
maxlen = 300
tRange = 30
pca_n = 5
def make_x_input_data (joint_dict, data):
    reldata = joint_dict.reldata("Head")
    pre_pca_data = []
    tRange_data = np.array([])
    tRange_data = np.append(tRange_data,[reldata.iloc[t//3,t%3] for t in range(tRange*3)])
    
    for tCount in range(0,reldata.shape[0]):
        if tCount+tRange>reldata.shape[0]:
            break
        pre_pca_data.append(tRange_data)  
        tRange_data = np.delete(tRange_data,(0,1,2),0)
        tRange_data = np.append(tRange_data,[reldata.iloc[tRange+tCount-1,t%3] for t in range(3)])
    print("ranged Head")    
    
    pca = PCA(pca_n)
    pca.fit(pre_pca_data)
    print(pca.explained_variance_ratio_)
    pca_data=pca.transform(pre_pca_data)
    
    tau_data=[]
    maxlen_data = np.array([pca_data[t//pca_n,t%pca_n] for t in range(maxlen*pca_n)])
    for cnt in range(0,pca_data.shape[0]):
        if cnt+maxlen > pca_data.shape[0]:
            break
        maxlen_data = np.reshape(maxlen_data,(maxlen,pca_n))
        maxlen_data = np.delete(maxlen_data,0,0)
        maxlen_data = np.append(maxlen_data,[pca_data[maxlen+cnt-1,t%5] for t in range(pca_n)])
        tau_data.append(maxlen_data)
    tau_data = np.reshape(tau_data,(len(tau_data),maxlen,pca_n))
    data.extend(tau_data)

    """
        tmp = np.zeros(len(x[0]))
        tmp2 = np.array([])
        for j in range(1, maxlen + 1):
            tmp += x[start + cnt + j - 1] / (maxlen // timelen)
            if j % (maxlen // timelen) == 0:
                tmp2 = np.append(tmp2, np.array(tmp))
                tmp = np.zeros(len(x[0]))
        tmp2 = np.reshape(tmp2, (timelen, 10))
        data.append(tmp2)
        target.append(y[start + cnt + maxlen])
        """

def make_y_input_data(joint_dict,threshold,target):
    time= np.array(joint_dict.joint_dict["Head"].time[:])
    idx = np.abs(np.asarray(time - threshold - time[0])).argmin()
    arr = np.zeros(len(time)-tRange+1-maxlen+1)
    for i in range(idx,len(time)-tRange+1-maxlen+1):
        arr[i] = 1
    target.extend(arr)
    
data = []
target = []
X_test = []
Y_test = []
g1 = body("G")
e1 = body("E")
h1 = body("H")

make_x_input_data(g1,data)
make_x_input_data(e1,data)
make_x_input_data(h1,X_test)
make_y_input_data(g1,660,target)
make_y_input_data(e1,280,target)
make_y_input_data(h1,230,Y_test)

X_train = np.array(data).reshape(len(data), maxlen, 5)  # データ数,時系列データ長さ,次元数
Y_train = np.array(target).reshape(len(target), 1)
X_validation = np.array(X_test).reshape(len(X_test), maxlen, 5)
Y_validation = np.array(Y_test).reshape(len(Y_test), 1)


def weight_variable(shape):
    return K.truncated_normal(shape, stddev=0.01)


n_in = len(X_train[0][0])
n_out = len(Y_train[0])
print(n_in, n_out)
model = Sequential()
model.add(GRU(100,
              init=weight_variable,
              input_shape=(300, n_in),
              return_sequences=True))
model.add(GRU(100,
              init=weight_variable,
              input_shape=(300, n_in),
              return_sequences=True))
model.add(GRU(50,
              init=weight_variable,
              input_shape=(300, n_in)))
model.add(BatchNormalization())
model.add(Dense(n_out, init=weight_variable))
model.add(Activation('sigmoid'))

optimizer = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999)
model.compile(loss='binary_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])

epochs = 50
batch_size = 10

hist = model.fit(X_train, Y_train,
                 batch_size=batch_size,
                 epochs=epochs,

                 validation_data=(X_validation, Y_validation))







######################     data_save      #################################





json_string = model.to_json()

open('train_GRU3.json', 'w').write(json_string)

model.save_weights("train_GRU3_param.hdf5")

model.save_weights("train_GRU3_param.h5")

# Accuracy

plt.plot(hist.history['acc'], label="accuracy")

plt.plot(hist.history['val_acc'], label="val_acc")

plt.title('model accuracy')

plt.xlabel('epoch')

plt.ylabel('accuracy')

plt.legend(loc="lower right")

# plt.show()

plt.savefig('Accuracy')

# loss

plt.clf()

plt.plot(hist.history['loss'], label="loss")

plt.plot(hist.history['val_loss'], label="val_loss")

plt.title('model loss')

plt.xlabel('epoch')

plt.ylabel('loss')

plt.legend(loc='lower right')

# plt.show()

plt.savefig('Loss')
