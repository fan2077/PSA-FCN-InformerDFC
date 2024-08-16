import os

import sys


import torch

from torch import nn
from torch.utils.data import Dataset

import scipy.io as sio
from tqdm import tqdm

import Informer_nodis


mat_label={"N":0,"Y":1}

class Matdataset(Dataset):
    def __init__(self,data_dir):
        self.lable_name={"N":0,"Y":1}
        self.data_info = self.get_mat_info(data_dir)

    def __getitem__(self, index):
        path_mat,label=self.data_info[index]

        matdata = sio.loadmat(path_mat)
        data = matdata['augmented_data1']
        mat = torch.Tensor(data)
        mat = mat.transpose(1, 0)

        return mat,label

    def __len__(self):
        return len(self.data_info)

    def get_mat_info(self,data_dir):
        data_info = list()
        for root, dirs, _ in os.walk(data_dir):
            # 遍历类别
            for sub_dir in dirs:
                mat_names = os.listdir(os.path.join(root, sub_dir))
                mat_names = list(filter(lambda x: x.endswith('.mat'), mat_names))
                # print(mat_names)
                # 遍历图片
                for i in range(len(mat_names)):
                    mat_name = mat_names[i]
                    path_mat = os.path.join(root, sub_dir, mat_name)
                    label = mat_label[sub_dir]
                    data_info.append((path_mat, int(label)))

        return data_info

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device2 = torch.device("cpu")

    data_root = "D:/DFCdata"  # get data root path
    mat_path = os.path.join(data_root, "DFC-40-6o")  # flower data set path
    assert os.path.exists(mat_path), "{} path does not exist.".format(mat_path)
    val_data=os.path.join(mat_path,'train')
    val_dataset=Matdataset(val_data)
    val_num=len(val_dataset)



    batch_size=1
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    #
    ASD_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                               batch_size=batch_size, shuffle=False,
                                               num_workers=nw)

    totel_ASD_mat = torch.zeros((6670,6670)).to(device)
    totel_NC_mat = torch.zeros((6670,6670)).to(device)
    for i in range(1,10):
        net =Informer_nodis.Informerencoder().to(device)
        # load model weights
        model_weight_path = "Informer40-6-2.pth".format(i)
        print(model_weight_path)
        net.load_state_dict(torch.load(model_weight_path, map_location=device))
        net.linear=nn.Sequential()
        net.eval()
        countASD=0
        countNC=0
        ASD_mat = torch.zeros((6670, 6670))
        NC_mat = torch.zeros((6670, 6670))
        with torch.no_grad():
            # predict class
            ASD_bar = tqdm(ASD_loader, file=sys.stdout)
            for step,data in enumerate(ASD_bar):
                mats,labels= data
                outputs,attn =net(mats.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                attn_0=attn[2]
                attn_0=torch.squeeze(attn_0)
                attn_0=attn_0[0]+attn_0[1]
                if labels[0]==0 and predict_y[0]==0 :
                    ASD_mat=ASD_mat.to(device)+attn_0
                    countASD+=1
                if labels[0]==1 and predict_y[0]==1:
                    NC_mat = NC_mat.to(device)+attn_0
                    countNC+=1
        # print(ASD_mat,NC_mat)
            print(countASD, countNC)
            ASD_mat=ASD_mat/countASD
            NC_mat=NC_mat/countNC
        totel_ASD_mat+=ASD_mat
        totel_NC_mat+=NC_mat

    totel_ASD_mat=(totel_ASD_mat/10.0).to(device2).numpy()
    totel_NC_mat=(totel_NC_mat/10.0).to(device2).numpy()
    sio.savemat("bioasd6670.mat", mdict={'data':totel_ASD_mat})
    sio.savemat("bionc6670.mat", mdict={'data':totel_NC_mat})
# ASD=sio.loadmat('bioasd.mat')
# NC=sio.loadmat('bionc.mat')
# totel_ASD_mat=ASD['data']
# totel_NC_mat=NC['data']
#
# c=abs(totel_ASD_mat-totel_NC_mat)
# listi=[]
# for i in range(116):
#     for j in range(116):
#         if c[i][j]>0.8:
#             listi.append(j)
#             listi.append(i)
#                 # listj.append(j)
#     # c=(c.to(device2)).numpy()
#     # print(np.nonzero(c))
#     # print(c)
# print(listi)
# listi_result=pandas.value_counts(listi)
# np.savetxt('bio.txt',listi_result,fmt='%d')
#     # listj_result=pandas.value_counts(listj)
#     # for i in range(10):
# # pandas.set_option('max_colwidth',100)
# print(listi_result.to_numpy())
# print(listi_result.index)







if __name__ == '__main__':
    main()
