import os
import scipy.io as scio
import numpy as np

file_path_data = 'E:\pycharmproject\pytorch1\preprocessed_data\data'
files_data = os.listdir(file_path_data)
Data = []
for data_name in files_data:
    loader_data_dict = scio.loadmat(file_path_data + '/' + data_name)
    loader_data = np.array(loader_data_dict['data']).transpose(2, 0, 1)
    Data.append(loader_data)

for idx in range(len(Data)):
    data = Data[idx]
    subject = ['A','A','B','B','B','B','C','C','E','E','E','F','F','F','G','G','H','I','I']
    b,c,t = data.shape
    var_data = data[:,:,1:] - data[:,:,0:169]
    print(var_data.max())
    varied_data = np.concatenate((np.zeros(shape=(b,c,1)),var_data),axis=2)
    np.save('../preprocessed_data/varying_data/'+str(idx+1)+'_Subject'+subject[idx]+'_data.npy',varied_data)

