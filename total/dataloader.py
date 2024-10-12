import os
import os.path as path
import scipy.io as scio
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.model_selection import KFold

class EEGData(Dataset):

    def __init__(self, x_data, x_label):
        self.len = len(x_data)
        self.data = x_data
        self.label = x_label

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return self.len

def makedir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)

def getdata_inside_subject(K=5):
    KF = KFold(n_splits=K,shuffle=True,random_state=0)
    file_path_data = 'E:\pycharmproject\pytorch1\preprocessed_data\data'
    file_path_label = 'E:\pycharmproject\pytorch1\preprocessed_data\label'
    files_data = os.listdir(file_path_data)
    files_label = os.listdir(file_path_label)
    Data = []
    Label = []
    R_trainDataset = []
    R_testDataset = []
    for data_name in files_data:
        loader_data_dict = scio.loadmat(file_path_data + '/' + data_name)
        loader_data = torch.FloatTensor(loader_data_dict['data']).permute(2, 0, 1)
        Data.append(loader_data)
        # print(loader_data.shape)

    for label_name in files_label:
        loader_label_dict = scio.loadmat(file_path_label + '/' + label_name)
        loader_label = torch.Tensor(loader_label_dict['label'])
        Label.append(loader_label)
        # print(loader_label.shape)

    for index in range(len(Data)):
        X = Data[index]
        Y = Label[index]
        for train_index, test_index in KF.split(X):

            X_train, X_test = X[train_index, :, :], X[test_index, :, :]
            Y_train, Y_test = Y[train_index], Y[test_index]

            Train_set = EEGData(X_train, Y_train)
            Test_set = EEGData(X_test, Y_test)

            R_trainDataset.append(Train_set)
            R_testDataset.append(Test_set)

    return R_trainDataset,R_testDataset

def getdata_cross_subject(K=5):
    KF = KFold(n_splits=K,shuffle=True,random_state=0)
    file_path_data = 'E:\pycharmproject\pytorch1\preprocessed_data\data'
    file_path_label = 'E:\pycharmproject\pytorch1\preprocessed_data\label'
    files_data = os.listdir(file_path_data)
    files_label = os.listdir(file_path_label)
    Data = []
    Label = []
    R_trainDataset = []
    R_testDataset = []
    for data_name in files_data:
        loader_data_dict = scio.loadmat(file_path_data + '/' + data_name)
        loader_data = torch.FloatTensor(loader_data_dict['data']).permute(2, 0, 1)
        Data.append(loader_data)
        # print(loader_data.shape)

    for label_name in files_label:
        loader_label_dict = scio.loadmat(file_path_label + '/' + label_name)
        loader_label = torch.Tensor(loader_label_dict['label'])
        Label.append(loader_label)
        # print(loader_label.shape)

    Dataline = None
    Labelline =None
    for index in range(len(Data)):
        if index == 0:
            Dataline = Data[index]
            Labelline = Label[index]
        else:
            Dataline = torch.cat([Dataline,Data[index]],0)
            Labelline = torch.cat([Labelline,Label[index]],0)

    for train_index, test_index in KF.split(Dataline):
        X_train, X_test = Dataline[train_index, :, :], Dataline[test_index, :, :]
        Y_train, Y_test = Labelline[train_index], Labelline[test_index]

        Train_set = EEGData(X_train, Y_train)
        Test_set = EEGData(X_test, Y_test)

        R_trainDataset.append(Train_set)
        R_testDataset.append(Test_set)

    return R_trainDataset,R_testDataset

if __name__ == '__main__':
    x_train,x_test = getdata_cross_subject()
    print(len(x_train))