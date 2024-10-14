import time
import torch
from torch.utils.data import dataloader
from model import EEGNet
from dataloader import getdata_cross_subject,getdata_inside_subject
import torch.nn as nn
import matplotlib.pyplot as plt

def train_inside_subject(k_fold_num,model,train_epoch,batch_size,subject_num,device,learning_rate,offset=False,offset_num=0,offset_step=1):
    #data_loader
    train_data, test_data = getdata_inside_subject(k_fold_num,offset=offset,offset_num=offset_num,offset_step=offset_step)
    Train_loss_subject = []
    Train_acc_subject = []
    Test_acc_subject = []

    for i in range(subject_num):
        one_subject_loss = 0.0
        one_subject_acc = 0.0
        one_subject_test_acc = 0.0
        for j in range(k_fold_num):
            model = model.to(device)
            train_data_subject = train_data[i*k_fold_num+j]
            test_data_subject = test_data[i*k_fold_num+j]
            train_data_loader = dataloader.DataLoader(
                dataset=train_data_subject,
                shuffle=True,
                batch_size=batch_size,
            )
            test_data_loader = dataloader.DataLoader(
                dataset=test_data_subject,
                shuffle=True,
            )
            train_name = 'subject-'+str(i+1)+'-k_fold-'+str(j+1)
            now_model = model
            now_model.reset_param()
            criterion = nn.CrossEntropyLoss().to(device)
            optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate,)
            r_loss,r_acc,t_acc =train_model(model=now_model,
                                     criterion=criterion,
                                     optimizer=optimizer,
                                     train_data_loader=train_data_loader,
                                     test_data_loader=test_data_loader,
                                     epoch_num=train_epoch,
                                     batch_size=batch_size,
                                     train_name=train_name,
                                     device=device,
                                     )
            one_subject_loss += r_loss
            one_subject_acc += r_acc
            one_subject_test_acc += t_acc

        one_subject_acc /=k_fold_num
        one_subject_loss /=k_fold_num
        one_subject_test_acc /=k_fold_num

        Train_acc_subject.append(one_subject_acc.item())
        Train_loss_subject.append(one_subject_loss)
        Test_acc_subject.append(one_subject_test_acc.item())

    for i in range(len(Train_loss_subject)):
        print('train_subject'+str(i+1)+'_acc:'+str(Train_acc_subject[i])+
              ' train_subject'+str(i+1)+'_loss:'+str(Train_loss_subject[i])+
              ' test_subject'+str(i+1)+'_acc:'+str(Test_acc_subject[i]))


    show_result(Train_acc_subject,Test_acc_subject)

def train_cross_subject(k_fold_num,model,train_epoch,batch_size,device,learning_rate,offset=False,offset_num=0,offset_step=1):
    train_data, test_data = getdata_cross_subject(k_fold_num,offset=offset,offset_num=offset_num,offset_step=offset_step)
    Train_loss = 0
    Train_acc = 0
    test_acc =0
    for j in range(k_fold_num):
        model = model.to(device)
        train_data_subject = train_data[j]
        test_data_subject = test_data[j]
        train_data_loader = dataloader.DataLoader(
            dataset=train_data_subject,
            shuffle=True,
            batch_size=batch_size,
        )
        test_data_loader = dataloader.DataLoader(
            dataset=test_data_subject,
            shuffle=True,
        )
        train_name = 'k_fold-' + str(j + 1)
        now_model = model
        now_model.reset_param()
        criterion = nn.CrossEntropyLoss().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        r_loss, r_acc, t_acc = train_model(model=now_model,
                                    criterion=criterion,
                                    optimizer=optimizer,
                                    train_data_loader=train_data_loader,
                                    test_data_loader=test_data_loader,
                                    epoch_num=train_epoch,
                                    batch_size=batch_size,
                                    train_name=train_name,
                                    device=device,
                                    )
        Train_loss += r_loss
        Train_acc += r_acc
        test_acc += t_acc
    Train_loss /=k_fold_num
    Train_acc /=k_fold_num
    test_acc /=k_fold_num

    print('train_acc:'+str(Train_acc)+' train_loss:'+str(Train_loss)+' test_acc:'+str(test_acc))


def train_model(model, criterion, optimizer , train_data_loader, test_data_loader, epoch_num, batch_size, train_name,device):
    train_loss = []
    train_acc = []
    test_acc=0
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

    print('---------------------Testing for '+train_name+'----------------------')
    model.eval()
    num =0
    for idx, (data_x, data_y) in enumerate(test_data_loader, 0):
        input = data_x.to(device)
        label = data_y.type(torch.LongTensor)
        label = label.to(device)
        label -= 1
        output = model(input)
        test_acc += sum(output.max(axis=1)[1] == label)
        num += len(label)
    print('test_acc:'+str(test_acc/num*100))


    return train_loss[epoch_num-1],train_acc[epoch_num-1],test_acc/num*100

def show_result(train_acc:torch.Tensor,test_acc):
    subject_name =[]
    for index in range(len(train_acc)):
        subject_name.append('subject_'+str(index))
    plt.plot(subject_name,train_acc)
    plt.plot(subject_name,test_acc)
    plt.xlabel('subject_num')
    plt.title('result')
    plt.ylabel('accuracy of the ex')
    plt.legend(['train','test'])
    plt.show()


if __name__ =="__main__":
    K_fold_num = 5
    batch_size = 80
    learning_rate = 0.0001
    Channel = 21
    Time_length = 170
    subject_num = 1
    train_epoch = 50
    device = 'cpu'
    model = EEGNet()
    print('give the mode')
    inputs = input()
    if(inputs=='inside'):
        train_inside_subject(
            k_fold_num=K_fold_num,
            model=model,
            train_epoch=train_epoch,
            batch_size=batch_size,
            subject_num=subject_num,
            device=device,
            learning_rate=learning_rate,
            offset=True,
            offset_step=2,
            offset_num=5
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