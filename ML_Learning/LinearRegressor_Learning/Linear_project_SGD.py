'''
案例：演示 随机梯度 线性回归对象 完成 波士顿房价预测案例。
回顾：线性回归算法 属于 有监督学习之 有特征，有标签，且标签是连续的。
    线性回归分类：
        一元线性回归：1 个特征列，1 个标签列。
        多元线性回归：多个特征列，1 个标签列。
    线性回归大白话解释：它是用线性公式来描述 特征 和 标签之间关系的，方便做预测，公式如下：
        一元线性回归：y=w∗x+b
        多元线性回归：
    如何衡量线性回归模型的好坏？
        思路：
            预测值和真实值之间的误差，误差越小，模型越好 ⇒ 损失函数
        具体的方案：
            最小二乘.　　　　每个 (样本) 误差平方和
            均方误差 (MSE)　每个 (样本) 误差平方和 / 样本总数
            均方根误差 (RMSE) 每个 (样本) 误差平方和 / 样本总数 的平方根
            平均绝对误差 (MAE) 每个 (样本) 误差绝对值和 / 样本总数
    如何让损失函数最小？
        思路 1：梯度下降法. ⇒ 全梯度下降 (Full Gradient Descent, FGD)，随机梯度下降 (SGD)，小批量梯度下降 (推荐，Min-Batch)，随机平均梯度下降 (SAG)
        思路 2：正规方程法.
机器学习开发流程：
    1.加载数据
    2.数据的预处理
    3.特征工程（特征提取，特征预处理）
    4.模型训练
    5.模型预测
    6.模型评估
'''

#导包
#from sklearn.datasets import fetch_california_housing                #数据
from sklearn.preprocessing import StandardScaler        #特征处理
from sklearn.model_selection import train_test_split    #数据集划分
from sklearn.linear_model import LinearRegression       #正规方程的回归模型
from sklearn.linear_model import SGDRegressor           #梯度下降的回归模型
from sklearn.metrics import mean_squared_error ,root_mean_squared_error, mean_absolute_error        #均方差评估
from sklearn.linear_model import Ridge,RidgeCV 
    
import pandas as pd
import numpy as np
#1. 加载波士顿房价数据
raw_df = pd.read_csv('ML_Learning/LinearRegressor_Learning/BostonHousing.csv'  #填对应的文件地址
                     , sep=",", header=None)
data = raw_df.values[1:,:13]
target = raw_df.values[1:,-1]
data = data.astype(float)
target = target.astype(float)

# print(f'特征：{data.shape}')
# print(f'标签：{target.shape}')
# print(f'特征数据：{data[:5]}')
# print(f'标签数据：{target[:5]}')

#2. 数据预处理 切分 训练集 和 测试集
x_train,x_test,y_trian,y_test = train_test_split(data,target,test_size=0.2,random_state=23)

#3. 特征工程（特征提取，特征预处理）
#3.1 创建标准化对象
transfer = StandardScaler()
#3.2 对训练集进行标准化
x_train = transfer.fit_transform(x_train)   #训练加转换
x_test = transfer.transform(x_test)

#4. 模型训练
#4.1 创建线性回归 正规方程 模型对象
#参1：是否计算截距 
#参2：学习率模式 ->常量 不会发生改变
#参3：学习率
estimator = SGDRegressor(fit_intercept=True,learning_rate='constant',eta0=0.01)    
#4.2 模型训练
estimator.fit(x_train,y_trian)
#4.3 打印模型计算出的权重 偏置
print(f'权重：{estimator.coef_}')
print(f'偏置：{estimator.intercept_}')

#5. 模型预测
y_pre = estimator.predict(x_test)
print(f'预测结果为：{y_pre}')

#6. 模型评估
print(f'均方误差为：{mean_squared_error(y_test,y_pre)}')
print(f'均方跟误差为：{root_mean_squared_error(y_test,y_pre)}')
print(f'平均绝对误差为：{mean_absolute_error(y_test,y_pre)}')