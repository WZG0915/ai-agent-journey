"""
案例：
    演示 欠拟合，正好拟合，过拟合，L1正则化，L2正则化的 效果图。

回顾：
    欠拟合：     模型在训练集 和 测试集表现效果都不好。
    正好拟合：   模型在训练集 和 测试集表现效果都好。
    过拟合：     模型在训练集表现好，测试集表现不好。

过拟合，欠拟合解释：

    产生原因：

        欠拟合：模型简单。
        过拟合：模型复杂。

    解决方案：

        欠拟合：增加特征，从而增加 模型的复杂度。
        过拟合：减少模型复杂度，手动筛选 (减少) 特征，L1 和 L2 正则化。

L1 和 L2 正则化介绍：

    目的 / 思路：

        都是基于 惩罚系数 来修改 (特征列的) 权重的，惩罚系数越大，则修改力度就越大，对应的权

    区别：

        L1 正则化，可以实现让权重变为 0，从而达到 特征选择的目的。
        L2 正则化，只能让权重无限趋近于 0，但是不能为 0。
    大白话：

        我要去爬山，带了个小包，装了：登山杖，水，面包，衣服，雨伞，鞋子…… 发现包装不下了。
        L1 正则化：可以实现去掉一些不是必选的，例如：当天去，当天回，且天气晴朗 → 不带雨伞，鞋子，即：权重为 0
        L2 正则化：换一个非常非常大的包，还是那些物品，但是空间占用 (权重) 就变小了……


"""

#导包
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.preprocessing import StandardScaler        #特征处理
from sklearn.model_selection import train_test_split    #数据集划分
from sklearn.linear_model import LinearRegression       #正规方程的回归模型
from sklearn.linear_model import SGDRegressor           #梯度下降的回归模型
from sklearn.metrics import mean_squared_error ,root_mean_squared_error, mean_absolute_error        #均方差评估
from sklearn.linear_model import Lasso,Ridge,RidgeCV 

#1. 定义函数，模拟：欠拟合
def dm01_under_fitting():
    #1. 准备数据 
    #1.1.指定随机种子
    np.random.seed(23)
    #1.2. 随机生成x轴 100个数据， 模拟:特征
    x = np.random.uniform(-3,3,100)
    #1.3. 基于x轴数据，通过线性公式，生成y轴数据 模拟:标签
    y = 0.5*x**2 + x + 2 + np.random.normal(0,1,100)
    #1.4. 查看生成的x和y轴数据

    #2. 数据预处理，把x轴数据转换成1列数据
    X = x.reshape(-1,1)

    #3. 特征工程
    #4. 模型训练
    #4.1. 创建模型对象
    estimator = LinearRegression()
    #4.2. 模型训练
    estimator.fit(X,y)  #参1：处理后的特征

    #5. 模型预测
    y_pre = estimator.predict(X)

    #6. 模型评估
    print(f'均方误差：{mean_squared_error(y,y_pre)}')

    #7. 绘图
    plt.scatter(x,y)
    plt.plot(x,y_pre,color='red')
    plt.show()
#2. 定义函数，模拟：正好拟合
def dm02_just_fitting():
    #1. 准备数据 
    #1.1.指定随机种子
    np.random.seed(23)
    #1.2. 随机生成x轴 100个数据， 模拟:特征
    x = np.random.uniform(-3,3,100)
    #1.3. 基于x轴数据，通过线性公式，生成y轴数据 模拟:标签
    y = 0.5*x**2 + x + 2 + np.random.normal(0,1,100)
    #1.4. 查看生成的x和y轴数据

    #2. 数据预处理，把x轴数据转换成1列数据
    X = x.reshape(-1,1)
    #增加一列特征列，增加模型复杂度
    X2 = np.hstack([X, X**2])     #函数作用：水平拼接

    #3. 特征工程
    #4. 模型训练
    #4.1. 创建模型对象
    estimator = LinearRegression()
    #4.2. 模型训练
    estimator.fit(X2,y)  #参1：处理后的特征

    #5. 模型预测
    y_pre = estimator.predict(X2)

    #6. 模型评估
    print(f'均方误差：{mean_squared_error(y,y_pre)}')

    #7. 绘图
    plt.scatter(x,y)
    #np.sort(x):对x做排序，默认：升序
    #np.argsort(x)：对x做排序，返回排序后的索引
    plt.plot(np.sort(x),y_pre[np.argsort(x)],color='red')
    plt.show()
#3. 定义函数，模拟：过拟合
def dm03_over_fitting():
    #1. 准备数据 
    #1.1.指定随机种子
    np.random.seed(23)
    #1.2. 随机生成x轴 100个数据， 模拟:特征
    x = np.random.uniform(-3,3,100)
    #1.3. 基于x轴数据，通过线性公式，生成y轴数据 模拟:标签
    y = 0.5*x**2 + x + 2 + np.random.normal(0,1,100)
    #1.4. 查看生成的x和y轴数据

    #2. 数据预处理，把x轴数据转换成1列数据
    X = x.reshape(-1,1)
    #增加一列特征列，增加模型复杂度
    X3 = np.hstack([X,X**2 ,X**3,X**4,X**5,X**6,X**7,X**8,X**9])     #函数作用：水平拼接


    #3. 特征工程
    #4. 模型训练
    #4.1. 创建模型对象
    estimator = LinearRegression()
    #4.2. 模型训练
    estimator.fit(X3,y)  #参1：处理后的特征

    #5. 模型预测
    y_pre = estimator.predict(X3)

    #6. 模型评估
    print(f'均方误差：{mean_squared_error(y,y_pre)}')

    #7. 绘图
    plt.scatter(x,y)
    #np.sort(x):对x做排序，默认：升序
    #np.argsort(x)：对x做排序，返回排序后的索引
    plt.plot(np.sort(x),y_pre[np.argsort(x)],color='red')
    plt.show()
#4. 定义函数，模拟L1正则化
def dm04_l1_regularization():
    #1. 准备数据 
    #1.1.指定随机种子
    np.random.seed(23)
    #1.2. 随机生成x轴 100个数据， 模拟:特征
    x = np.random.uniform(-3,3,100)
    #1.3. 基于x轴数据，通过线性公式，生成y轴数据 模拟:标签
    y = 0.5*x**2 + x + 2 + np.random.normal(0,1,100)
    #1.4. 查看生成的x和y轴数据

    #2. 数据预处理，把x轴数据转换成1列数据
    X = x.reshape(-1,1)
    #增加一列特征列，增加模型复杂度
    X3 = np.hstack([X,X**2 ,X**3,X**4,X**5,X**6,X**7,X**8,X**9])     #函数作用：水平拼接

    #3. 特征工程
    #4. 模型训练
    #4.1. 创建模型对象
    #estimator = LinearRegression()
    #改为创建L1正则化对象
    estimator = Lasso(alpha=0.1)    #alpha：正则化系数
    
    #4.2. 模型训练
    estimator.fit(X3,y)  #参1：处理后的特征

    #5. 模型预测
    y_pre = estimator.predict(X3)

    #6. 模型评估
    print(f'均方误差：{mean_squared_error(y,y_pre)}')

    #7. 绘图
    plt.scatter(x,y)
    #np.sort(x):对x做排序，默认：升序
    #np.argsort(x)：对x做排序，返回排序后的索引
    plt.plot(np.sort(x),y_pre[np.argsort(x)],color='red')
    plt.show()
#5. 定义函数，模拟：L2正则化
def dm05_l2_regularization():
    #1. 准备数据 
    #1.1.指定随机种子
    np.random.seed(23)
    #1.2. 随机生成x轴 100个数据， 模拟:特征
    x = np.random.uniform(-3,3,100)
    #1.3. 基于x轴数据，通过线性公式，生成y轴数据 模拟:标签
    y = 0.5*x**2 + x + 2 + np.random.normal(0,1,100)
    #1.4. 查看生成的x和y轴数据

    #2. 数据预处理，把x轴数据转换成1列数据
    X = x.reshape(-1,1)
    #增加一列特征列，增加模型复杂度
    X3 = np.hstack([X,X**2 ,X**3,X**4,X**5,X**6,X**7,X**8,X**9])     #函数作用：水平拼接

    #3. 特征工程
    #4. 模型训练
    #4.1. 创建模型对象
    #estimator = LinearRegression()
    #改为创建L1正则化对象
    #estimator = Lasso(alpha=0.1)    #alpha：正则化系数
    #改为创建L2正则化对象
    estimator = Ridge(alpha=10)    #alpha：正则化系数    
    #4.2. 模型训练
    estimator.fit(X3,y)  #参1：处理后的特征

    #5. 模型预测
    y_pre = estimator.predict(X3)

    #6. 模型评估
    print(f'均方误差：{mean_squared_error(y,y_pre)}')

    #7. 绘图
    plt.scatter(x,y)
    #np.sort(x):对x做排序，默认：升序
    #np.argsort(x)：对x做排序，返回排序后的索引
    plt.plot(np.sort(x),y_pre[np.argsort(x)],color='red')
    plt.show()
#6. 测试
if __name__ == '__main__':
    dm05_l2_regularization()