#导包
import os
os.environ['OMP_NUM_THREADS'] = '4'


from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.metrics import calinski_harabasz_score,silhouette_score

#1. 定义函数，演示：SSE + 肘部法
def dm01_sse():
    #1. 定义ees列表，记录：每个k值的SSE值
    sse_list = []
    #2. 生成数据集
    x,y = make_blobs(
        n_samples= 1000,
        n_features=2,
        centers=[[-1,-1],[0,0],[1,1],[2,2]],
        cluster_std=[0.4,0.2,0.2,0.2],
        random_state=23
    )
    #3. for训练遍历，获取每个K值，计算对应的sse值，并添加到sse_list中
    for k in range(1,100):
        #3.1 创建KMeans对象，指定K值，迭代次数，随机种子
        estimator = KMeans(n_clusters=k,max_iter=100,random_state=23)
        #3.2 训练模型
        estimator.fit(x)
        #3.3 模型预测
        #3.4 获取每个簇的sse的值
        sse_value = estimator.inertia_
        #3.5 将每个k值对应的sse值添加到sse_list中
        sse_list.append(sse_value)
    #4. 绘制sse曲线 -> 数据的可视化
    plt.figure(figsize=(20,10))
    plt.xticks(range(0,100,3))
    #参1：k值 参2：k值对应的sse值
    plt.plot(range(1,100),sse_list,marker='o')
    plt.show()

#2. 定义函数，演示：SC轮廓系数法
def dm02_sc():
    #1. 定义ees列表，记录：每个k值的SSE值
    sc_list = []
    #2. 生成数据集
    x,y = make_blobs(
        n_samples= 1000,
        n_features=2,
        centers=[[-1,-1],[0,0],[1,1],[2,2]],
        cluster_std=[0.4,0.2,0.2,0.2],
        random_state=23
    )
    #3. for训练遍历，获取每个K值，计算对应的sse值，并添加到sse_list中
    for k in range(2,100):  #考虑簇外
        #3.1 创建KMeans对象，指定K值，迭代次数，随机种子
        estimator = KMeans(n_clusters=k,max_iter=100,random_state=23)
        #3.2 训练模型
        y_pre = estimator.fit_predict(x)
        #3.3 模型预测
        #3.4 获取每个簇的sse的值
        sc_value =  silhouette_score(x,y_pre)
        #3.5 将每个k值对应的sse值添加到sse_list中
        sc_list.append(sc_value)
    #4. 绘制sse曲线 -> 数据的可视化
    plt.figure(figsize=(20,10))
    plt.xticks(range(0,100,3))
    #参1：k值 参2：k值对应的sse值
    plt.plot(range(2,100),sc_list,marker='o')
    plt.show()
#3. 定义函数，演示：CH轮廓系数法
def dm03_ch():
    #1. 定义ees列表，记录：每个k值的SSE值
    ch_list = []
    #2. 生成数据集
    x,y = make_blobs(
        n_samples= 1000,
        n_features=2,
        centers=[[-1,-1],[0,0],[1,1],[2,2]],
        cluster_std=[0.4,0.2,0.2,0.2],
        random_state=23
    )
    #3. for训练遍历，获取每个K值，计算对应的sse值，并添加到sse_list中
    for k in range(2,100):  #考虑簇外
        #3.1 创建KMeans对象，指定K值，迭代次数，随机种子
        estimator = KMeans(n_clusters=k,max_iter=100,random_state=23)
        #3.2 训练模型
        y_pre = estimator.fit_predict(x)
        #3.3 模型预测
        #3.4 获取每个簇的sse的值
        ch_value =  calinski_harabasz_score(x,y_pre)
        #3.5 将每个k值对应的sse值添加到sse_list中
        ch_list.append(ch_value)
    #4. 绘制sse曲线 -> 数据的可视化
    plt.figure(figsize=(20,10))
    plt.xticks(range(0,100,3))
    #参1：k值 参2：k值对应的sse值
    plt.plot(range(2,100),ch_list,marker='o')
    plt.show()

#4. 测试
if __name__ == "__main__":
    dm03_ch()
