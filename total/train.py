import time
import torch
from torch.utils.data import dataloader,dataset
from model import EEGNet
from dataloader import getdata_cross_subject,getdata_inside_subject
import torch.nn as nn

def train_inside_subject(k_fold_num,model,train_epoch,batch_size,subject_num,device,learning_rate):
    #data_loader
    train_data, _ = getdata_inside_subject(k_fold_num)
    print(len(train_data))
    Train_loss_subject = []
    Train_acc_subject = []

    for i in range(subject_num):
        one_subject_loss = 0
        one_subject_acc = 0
        for j in range(k_fold_num):
            model = model.to(device)
            train_data_subject = train_data[i*k_fold_num+j]
            train_data_loader = dataloader.DataLoader(
                dataset=train_data_subject,
                shuffle=True,
                batch_size=batch_size,
            )
            train_name = 'subject-'+str(i+1)+'-k_fold-'+str(j+1)
            now_model = model
            now_model.reset_param()
            criterion = nn.CrossEntropyLoss().to(device)
            optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
            r_loss,r_acc=train_model(model=now_model,
                                     criterion=criterion,
                                     optimizer=optimizer,
                                     train_data_loader=train_data_loader,
                                     epoch_num=train_epoch,
                                     batch_size=batch_size,
                                     train_name=train_name,
                                     device=device,
                                     )
            one_subject_loss += r_loss
            one_subject_acc += r_acc

        one_subject_acc /=k_fold_num
        one_subject_loss /=k_fold_num

        Train_acc_subject.append(one_subject_acc)
        Train_loss_subject.append(one_subject_loss)

    for i in range(len(Train_loss_subject)):
        print('subject'+str(i)+'_acc:'+str(Train_acc_subject[i])+' subject'+str(i)+'_loss:'+str(Train_loss_subject[i]))

def train_cross_subject(k_fold_num,model,train_epoch,batch_size,device,learning_rate):
    train_data, _ = getdata_cross_subject(k_fold_num)
    print(len(train_data))
    Train_loss = 0
    Train_acc = 0
    for j in range(k_fold_num):
        model = model.to(device)
        train_data_subject = train_data[j]
        train_data_loader = dataloader.DataLoader(
            dataset=train_data_subject,
            shuffle=True,
            batch_size=batch_size,
        )
        train_name = 'k_fold-' + str(j + 1)
        now_model = model
        now_model.reset_param()
        criterion = nn.CrossEntropyLoss().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        r_loss, r_acc = train_model(model=now_model,
                                    criterion=criterion,
                                    optimizer=optimizer,
                                    train_data_loader=train_data_loader,
                                    epoch_num=train_epoch,
                                    batch_size=batch_size,
                                    train_name=train_name,
                                    device=device,
                                    )
        Train_loss += r_loss
        Train_acc += r_acc
    Train_loss /=k_fold_num
    Train_acc /=k_fold_num

    print('acc:'+str(Train_acc)+' loss:'+str(Train_loss))

def train_model(model, criterion, optimizer , train_data_loader, epoch_num, batch_size, train_name,device):
    train_loss = []
    train_acc = []
    print('---------------------'+train_name+'---------------------')
    for turn in range(train_epoch):
        print('---------------------Training(epoch: %d )----------------------' % (turn + 1))
        t_s = time.time()
        model.train()
        model.to(device)
        acc, num = 0, 0
        running_loss = 0
        for idx, (data_x, data_y) in enumerate(train_data_loader, 0):
            input = data_x.to(device)
            label = data_y.type(torch.LongTensor)
            label = label.to(device)
            if len(label) < batch_size:
                label = label.view(len(label))
            else:
                label = label.view(batch_size)
            label -=1
            optimizer.zero_grad()
            output = model(input)
            loss = criterion(output,label)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            acc += sum(output.max(axis=1)[1] == label)
            num += len(label)
        print('loss:'+str(running_loss/len(train_data_loader))+' acc:'+str(acc/num*100)+'%')
        train_loss.append(running_loss/len(train_data_loader))
        train_acc.append(acc/num*100)

    return train_loss[epoch_num-1],train_acc[epoch_num-1],



if __name__ =="__main__":
    K_fold_num = 5
    batch_size = 32
    learning_rate = 0.0001
    Channel = 21
    Time_length = 170
    subject_num = 19
    train_epoch = 50
    device = 'cpu'
    model = EEGNet()
    inputs = input()
    if(inputs=='inside'):
        train_inside_subject(
            k_fold_num=K_fold_num,
            model=model,
            train_epoch=train_epoch,
            batch_size=batch_size,
            subject_num=subject_num,
            device=device,
            learning_rate=learning_rate
        )
    else:
        train_cross_subject(
            k_fold_num=K_fold_num,
            model=model,
            train_epoch=train_epoch,
            batch_size=batch_size,
            device=device,
            learning_rate=learning_rate
        )