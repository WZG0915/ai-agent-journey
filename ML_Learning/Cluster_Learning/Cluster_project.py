#导包
import os
os.environ['OMP_NUM_THREADS'] = '4'


from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import calinski_harabasz_score,silhouette_score
import pandas as pd

#1. 定义函数，找聚类的 质心数
def dm01_find_k():
    #1. 加载数据集
    df = pd.read_csv('E:\\To_learn_Ai\\ML_Learning\\Cluster_Learning\\data\\customers.csv')

    #2. 定义sse_list,sc_list 记录：不同k值的 评估效果
    sse_list = []       #sse只考虑簇内，越小越好
    sc_list = []        #sc考虑簇内和簇间，越大越好
    #特征提取
    x = df.iloc[:,3:5]

    #3. 定义for循环，测试不同k值的 评估效果
    for k in range(2,20):
        #4. 创建模型对象
        estimator = KMeans(n_clusters=k,max_iter=100,random_state=23)
        #5. 模型训练
        estimator.fit(x)
        #6. 模型预测
        y_pre = estimator.predict(x)
        #7. 分别把评分增加到对应的列表中
        sse_list.append(estimator.inertia_)
        sc_list.append(silhouette_score(x,y_pre)) 
    #4. 绘制折线图，看k值哪个效果好
    plt.figure(figsize=(20,10))
    plt.plot(range(2,20),sse_list)
    plt.show()
    plt.plot(range(2,20),sc_list)
    plt.show()
#2. 定义函数，实现：模型训练，模型预测，模型评估
def dm02_train_predict_evaluate():
    #1. 加载数据集
    df = pd.read_csv('E:\\To_learn_Ai\\ML_Learning\\Cluster_Learning\\data\\customers.csv')

    #2. 定义sse_list,sc_list 记录：不同k值的 评估效果
    sse_list = []       #sse只考虑簇内，越小越好
    sc_list = []        #sc考虑簇内和簇间，越大越好
    #特征提取
    x = df.iloc[:,3:5]
    #3. 模型训练，k=5 dm01得到的
    estimator = KMeans(n_clusters=5,max_iter=100,random_state=23)
    estimator.fit(x)
    #4. 模型预测
    y_pre = estimator.predict(x)
    #5. 绘制5个簇的样本点
    plt.scatter(x.values[y_pre==0,0], x.values[y_pre==0,1])
    plt.scatter(x.values[y_pre==1,0], x.values[y_pre==1,1])
    plt.scatter(x.values[y_pre==2,0], x.values[y_pre==2,1])
    plt.scatter(x.values[y_pre==3,0], x.values[y_pre==3,1])
    plt.scatter(x.values[y_pre==4,0], x.values[y_pre==4,1])
    #6. 绘制5个簇的质心点
    plt.scatter(estimator.cluster_centers_[:,0],estimator.cluster_centers_[:,1],c='black')

    plt.show()

#3. 测试
if __name__ == "__main__":
    dm02_train_predict_evaluate()
