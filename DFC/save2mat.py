import os
import numpy as np
import pandas as pd
from scipy.io import savemat


# 读取1D文件数据
# 读取1D文件数据
def read_1d_file(file_path):
    matdata = pd.read_table(file_path, header=None)
    data = matdata.values
    return data

# 保存为MAT文件
def save_as_mat(data, mat_file_path):
    savemat(mat_file_path, {'value': data})


# 读取指定目录下所有 .1D 文件并保存为 .mat 文件
def process_all_1d_files(directory):
    for filename in os.listdir(directory):
        if filename.endswith('.1D'):
            file_path = os.path.join(directory, filename)
            mat_file_path = os.path.join(directory, filename.replace('.1D', '.mat'))

            data = read_1d_file(file_path)
            save_as_mat(data, mat_file_path)

            print(f"{filename} 已处理并保存为 {mat_file_path}")


# 示例用法
input_directory = r'F:\data_set\ABIDE\ABIDE\Y'  # 替换为你存放 .1D 文件的目录路径
process_all_1d_files(input_directory)
