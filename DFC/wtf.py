import os
import numpy as np
import torch
import scipy.io as sio
from torch.utils.data import Dataset
from torchvision.transforms import transforms

W=40
S=5
numwidows=(145-W)//S+1
mat_label={"N":0,"Y":1}

def sliding_window(data, w, s):
    """使用滑动窗法切割数据"""
    num_windows = (data.shape[0] - w) // s + 1  # 计算切割出的窗口数
    windows = np.zeros((num_windows, 116, data.shape[1]))  # 初始化窗口数组
    for i in range(num_windows):
        window_data = data[i * s:i * s + w]  # 切割窗口
        pearson_coef = np.corrcoef(window_data.T)
        pearson_coef = np.corrcoef(pearson_coef)
        pearson_coef=np.clip(pearson_coef,-0.999,0.999)
        z_matrix=np.arctanh(pearson_coef)
        windows[i]=z_matrix
    return windows

def flatten_upper_triangular(windows):
    num_windows, w, _ = windows.shape
    flattened = np.zeros((num_windows, w * (w - 1) // 2))

    for i in range(num_windows):
        upper_triangular = np.triu(windows[i], k=1)  # 取上三角部分
        flattened[i] = upper_triangular[np.nonzero(upper_triangular)]  # 取非零元素，展开成一维数组

    return flattened

def roll_matrix(matrix):
    """
    将矩阵的第一行移到最后一行，实现数据的扩充操作
    参数：
        - matrix: numpy 数组，待操作的矩阵
    返回：
        - numpy 数组，经过数据扩充操作后的矩阵
    """
    # 使用 np.roll 函数将第一行移到最后一行

    rolled_matrix = np.roll(matrix, shift=-1, axis=0)

    return rolled_matrix


def custom_transform(mat):
    rows = mat.shape[0]
    # 创建一个空的 numpy 数组，用于存储生成的多个新数据
    rolled_mats = []

    # 遍历矩阵的每一行
    for i in range(rows):
        # 调用 roll_matrix 函数对当前行进行循环移动
        rolled_mat = roll_matrix(mat)
        # 将移动后的行添加到 rolled_mats 列表中
        rolled_mats.append(rolled_mat)
        mat=rolled_mat

    return rolled_mats


# 将自定义的数据变换函数封装为一个类
class CustomTransform(object):
    def __call__(self, mat):
        return custom_transform(mat)

data_transforms = transforms.Compose([
    CustomTransform()
])


import os
import scipy.io as sio
import numpy as np

w=str(W)
s=str(S)
b="DFC-"+w+"-"+s
a="N"

# 定义输入路径和输出路径
# input_path = r"F:\data_set\ABIDE\ABIDE"  # 输入路径
# output_path = "D:/DFCdata"  # 输出路径
#
# input_path = os.path.join(input_path, a)
# output_path=os.path.join(output_path,b,"train",a)
#
#
# # 获取输入路径下的所有.mat文件
# mat_files = [f for f in os.listdir(input_path) if f.endswith(".mat")]
#
# # 循环读取每个.mat文件，并进行数据扩充处理
# for i, mat_file in enumerate(mat_files):
#     # 读取.mat文件
#     mat_data = sio.loadmat(os.path.join(input_path, mat_file))
#     # print(mat_data)
#     data = mat_data['value']
#     data = data[1:, :]
#     data = np.array(data, dtype=float)
#     data = torch.Tensor(data)
#     windows = sliding_window(data, W, S)
#     flattened = flatten_upper_triangular(windows)  # 展开上三角部分
#     mat = torch.Tensor(flattened)
#     # 进行数据扩充处理，这里只是简单示范，可以根据具体需求进行数据处理
#     augmented_mats = custom_transform(mat)
#
#     for j, augmented_item in enumerate(augmented_mats):
#         # 构建新文件名
#         # 按照序号命名新文件
#         new_file_name = f"augmented_data_{i + 1}_{j+1}.mat"
#         new_var_name = f"augmented_data1"
#          # 将新数据保存到输出路径下
#         sio.savemat(os.path.join(output_path, new_file_name), {new_var_name: augmented_item})
#
#     print(f"已处理文件: {mat_file}，生成新文件: {new_file_name}")
#
# print("数据处理完成1！")

#
# 定义输入路径和输出路径
input_path = r"D:\DFCdata\ABIDE\USM"  # 输入路径
output_path= "D:/DFCdata"  # 输出路径


input_path = os.path.join(input_path, a)
output_path=os.path.join(output_path,b,"val",a)

# 获取输入路径下的所有.mat文件
mat_files = [f for f in os.listdir(input_path) if f.endswith(".mat")]

# 循环读取每个.mat文件，并进行数据扩充处理
for i, mat_file in enumerate(mat_files):
    # 读取.mat文件
    mat_data = sio.loadmat(os.path.join(input_path, mat_file))
    data = mat_data['value']
    data = data[1:, :]
    data = np.array(data, dtype=float)
    data = torch.Tensor(data)
    windows = sliding_window(data, W, S)
    flattened = flatten_upper_triangular(windows)  # 展开上三角部分

    new_file_name = f"augmented_data_{i + 1}.mat"
    new_var_name = f"augmented_data1"
         # 将新数据保存到输出路径下
    sio.savemat(os.path.join(output_path, new_file_name), {new_var_name: flattened})

    print(f"已处理文件: {mat_file}，生成新文件: {new_file_name}")

print("数据处理完成2！")


