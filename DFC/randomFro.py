from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import os
import numpy as np
import scipy.io as sio
# 假设你的数据已经准备好，包括特征矩阵X和标签y
# X是形状为(92, 116, 116)的特征矩阵，y是包含92个类别标签的数组
# 指定存储数据的文件夹路径
data_folder = "C:/Users/Lenovo/Desktop/deep-learning-for-image-processing-master/data_set/rf"  # 修改为你的数据文件夹路径

# 初始化空的特征矩阵X和标签列表y
X = []
y = []

# 遍历文件夹中的每个子文件夹（每个子文件夹代表一个类别）
for label in os.listdir(data_folder):
    label_folder = os.path.join(data_folder, label)

    # 遍历每个子文件夹中的图像文件
    for image_file in os.listdir(label_folder):
        image_path = os.path.join(label_folder, image_file)

        # 读取图像并将其调整为适当的大小（这里使用了skimage库）
        image = sio.loadmat(image_path)


        # 将图像数据添加到特征矩阵X中
        X.append(image)

        # 将标签添加到标签列表y中
        y.append(label)

# 将X和y转换为NumPy数组
X = np.array(X)
y = np.array(y)


# 数据拆分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建随机森林分类器并训练模型
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

# 使用模型进行预测
y_pred = rf_classifier.predict(X_test)

# 评估模型性能
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

classification_rep = classification_report(y_test, y_pred)
print("Classification Report:\n", classification_rep)
