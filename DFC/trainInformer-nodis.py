import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import scipy.io as sio
from torch.utils.data import Dataset
import Informer_nodis
# import matplotlib.pyplot as plt
# import fullattentionDFC
# import nodistilling

mat_label={"N":0,"Y":1}


class Matdataset(Dataset):
    def __init__(self,data_dir):
        self.lable_name={"N":0,"Y":1}
        self.data_info = self.get_mat_info(data_dir)

    def __getitem__(self, index):
        path_mat,label=self.data_info[index]

        matdata = sio.loadmat(path_mat)
        data = matdata['augmented_data1']
        # data = matdata['data']
        # data=np.corrcoef(data)
        mat = torch.Tensor(data)
        mat=mat.transpose(1,0)
        # print(mat.shape)

        return mat, label

    def __len__(self):
        return len(self.data_info)

    def get_mat_info(self,data_dir):
        data_info = list()
        for root, dirs, _ in os.walk(data_dir):
            # 遍历类别
            for sub_dir in dirs:
                mat_names = os.listdir(os.path.join(root, sub_dir))
                mat_names = list(filter(lambda x: x.endswith('.mat'), mat_names))
                # 遍历mat
                for i in range(len(mat_names)):
                    mat_name = mat_names[i]
                    path_mat = os.path.join(root, sub_dir, mat_name)
                    label = mat_label[sub_dir]
                    data_info.append((path_mat, int(label)))

        return data_info


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    data_root = "D:/DFCdata"  # get data root path
    mat_path = os.path.join(data_root, "DFC-40-6")  # flower data set path
    assert os.path.exists(mat_path), "{} path does not exist.".format(mat_path)
    train_data_dir=os.path.join(mat_path,"train")
    val_data_dir=os.path.join(mat_path,"val")
    train_dataset=Matdataset(data_dir=train_data_dir)
    train_num = len(train_dataset)
    # print(train_dataset)
    # {'1':0, '-1':1}
    train_list = train_dataset.lable_name
    cla_dict = dict((val, key) for key, val in train_list.items()) #字典
    # write dict into json file
    json_str = json.dumps(cla_dict, indent=4)
    with open('./class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    batch_size = 1
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    #
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw)
    validate_dataset = Matdataset(val_data_dir)
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(dataset=validate_dataset,
                                                  batch_size=batch_size, shuffle=True,
                                                  num_workers=nw)
    #
    print("using {} mat for training, {} mat for validation.".format(train_num,
                                                                           val_num))



    lacc,lTPR,lTNR,lPrec,lF1=0.0,0.0,0.0,0.0,0.0
    num_epoch=1

    save_path='Informer40-6-2.pth'
    for i in range(num_epoch):
        print("第%d轮开始："%(i))
        net=Informer_nodis.Informerencoder().to(device)
        # net =fullattentionDFC.create_model2().to(device)
        # net = nodistilling.Informerencoder().to(device)
        loss_function = nn.CrossEntropyLoss()

        optimizer = optim.Adam(net.parameters(), lr=0.0001)
        epochs = 30
        best_acc = 0.0
        best_TPR=0.0
        best_TNR=0.0
        best_Prec=0.0
        best_F1=0.0
        train_steps = len(train_loader)
        train_losses=[]
        train_acces=[]
        val_losses=[]
        val_acces=[]
        for epoch in range(epochs):
            # train
            net.train()
            running_loss = 0.0
            accx=0.0
            train_acc=0.0
            train_bar = tqdm(train_loader, file=sys.stdout)
            for step, data in enumerate(train_bar):
                mats, labels = data
                optimizer.zero_grad()
                outputs,attn= net(mats.to(device))
                # outputs= net(mats.to(device))
                predict_x = torch.max(outputs, dim=1)[1]
                loss = loss_function(outputs, labels.to(device))
                loss.backward()
                optimizer.step()
                # print statistics
                running_loss += loss.item()
                accx += torch.eq(predict_x, labels.to(device)).sum().item()

                train_acc=accx/train_num
                train_acces.append(train_acc)
                train_losses.append(running_loss/train_num)
                train_bar.desc = "train epoch[{}/{}] train_acc:{:.3f} loss:{:.3f}".format(epoch + 1,
                                                                             epochs,train_acc,
                                                                         loss)

            # validate
            net.eval()
            acc = 0.0  # accumulate accurate number / epoch
            TP = 0
            FP = 0
            TN = 0
            FN = 0
            val_loss=0.0
            with torch.no_grad():
                val_bar = tqdm(validate_loader, file=sys.stdout)
                for val_data in val_bar:
                    val_mats, val_labels = val_data
                    # test.append(val_labels.astype('int32'))
                    outputs ,attn= net(val_mats.to(device))
                    # outputs= net(val_mats.to(device))
                    loss = loss_function(outputs, val_labels.to(device))
                    val_loss += loss.item()
                    predict_y = torch.max(outputs, dim=1)[1]
                    for i in range(batch_size):
                        if predict_y[i] == 0 and val_labels[i] == 0:  # 预测出来是0，真实标签也是0
                            TN += 1
                        if predict_y[i] == 0 and val_labels[i] == 1:  # 预测出来是0，真实标签是1
                            FN += 1
                        if predict_y[i] == 1 and val_labels[i] == 0:  # 预测出来是1，真实标签是0
                            FP += 1
                        if predict_y[i] == 1 and val_labels[i] == 1:  # 预测出来是1，真实标签是1
                            TP += 1
                    acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

            # if early_stopping(val_loss):  # 传入当前验证集损失值
            #     print('Early stopping at epoch:', epoch + 1)
            #     break
            print(TP,FP,TN,FN)
            val_accurate = acc / val_num
            TPR=TP / (TP + FN)  # 真正类率TPR recall 灵敏度
            # TPR=1 # 真正类率TPR recall 灵敏度
            # Prec=TP / (TP+FP)     #精准
            FNR=FN / (TP + FN)  # 假负类率FNR
            # FNR=1  # 假负类率FNR
            FPR=FP / (FP + TN)  # 假正类率FPR
            # FPR=1 # 假正类率FPR
            TNR=TN / (TN + FP)  # 真负类率TNR 特异度
            # TNR=1  # 真负类率TNR 特异度
            if (TP + FP) != 0:
                Prec = TP / (TP + FP)
                F1 = 2 * (1.0 / (1.0 / Prec + 1.0 / TPR))
            else:
                Prec = 0.0
                F1 = 0.0
            val_acces.append(val_accurate)
            val_losses.append(val_loss/val_num)
            print('[epoch %d] train_loss: %.3f train_acc: %.3f val_accuracy: %.3f val_loss:%.3f TPR: %.3f  FNR: %.3f  FPR: %.3f  TNR: %.3f  Prec: %.3f  F1: %.3f' %
                  (epoch + 1, running_loss / train_steps,train_acc, val_accurate, val_loss/val_num,TPR, FNR, FPR, TNR ,Prec ,F1 ))


            if val_accurate > best_acc:
                best_acc = val_accurate
                best_TPR=TPR
                best_TNR =TNR
                best_Prec=Prec
                best_F1=F1
                torch.save(net.state_dict(),save_path)
            # if best_acc>global_acc:
            #     torch.save(net.state_dict(), save_path)
            #     global_acc=best_acc
        lacc+=best_acc
        lTPR+=best_TPR
        lTNR+=best_TNR
        lPrec+=best_Prec
        lF1+=best_F1

        print("第%d大轮结束"%(i))
    print("5轮结果为 acc:%.3f TPR:%.3f TNR:%.3f Prec:%.3f F1:%.3f"%(lacc/num_epoch,lTPR/num_epoch,lTNR/num_epoch,lPrec/num_epoch,lF1/num_epoch))

    # plt.plot(np.arange(len(train_acces)),train_acces, label="train_acc")
    # # plt.plot(np.arange(len(train_losses)),train_losses, label="train_loss")
    # # plt.plot(np.arange(len(val_losses)),val_losses, label="val_loss")
    # plt.plot(np.arange(len(val_acces)), val_acces,label="val_acc")
    # plt.legend()
    # plt.xlabel('epoch')
    # plt.title('Model acc&loss')
    # # plt.savefig("DFC60_5.jpg")
    # plt.show()


if __name__ == '__main__':
    main()


