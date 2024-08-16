import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# 示例窗口数据，假设您的sliding_window函数返回了这些窗口数据
windows = np.array([
    # 窗口1数据
    [[0.1, 0.2, 0.3], [0.2, 0.3, 0.4], [0.3, 0.4, 0.5]],
    # 窗口2数据
    [[0.2, 0.3, 0.4], [0.3, 0.4, 0.5], [0.4, 0.5, 0.6]],
    # 添加更多窗口数据
])

# 计算余弦相似度矩阵
cosine_similarity_matrix = cosine_similarity(windows.reshape(windows.shape[0], -1).astype(np.float64))  # 修改这里

# 打印余弦相似度矩阵
print("余弦相似度矩阵：")
print(cosine_similarity_matrix)
