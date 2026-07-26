#导包
import os
os.environ['OMP_NUM_THREADS'] = '4'

from sklearn.cluster import KMeans          #聚类API，采用指定 质心 来分簇
import matplotlib.pyplot as plt             #绘图的
from sklearn.datasets import make_blobs     #默认按照高斯分布生成数据集，只需指定 均值 方差
from sklearn.metrics import calinski_harabasz_score



#1. 准备数据集
#参1：样本数量 参2：样本特征数量 参3：样本标签数量 参4：标准层 参5：随机种子
x,y = make_blobs(n_samples=1000, n_features=2, centers=3, cluster_std=1.0, random_state=0)
# print(x)
# print(y)

#2. 绘制上述的图形
#参1：横坐标 参2：纵坐标 参3：颜色
plt.scatter(x[:,0],x[:,1]) 
plt.show()

#3. 创建KMeans对象
#参1：聚类数量 参2：随机种子
estimator = KMeans(n_clusters=4, random_state=23)

#4. 模型训练 和 预测
y_pre = estimator.fit_predict(x)         #预测值
plt.scatter(x[:,0],x[:,1],c=y_pre)
plt.show()

#5. 评价指标
print(f'评价指标：{calinski_harabasz_score(x,y_pre)}')