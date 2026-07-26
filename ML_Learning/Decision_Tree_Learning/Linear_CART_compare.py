#导包
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

#1. 加载数据
x_train = np.array(list(range(1,11))).reshape(-1,1)
y_train = np.array([5.56, 5.7, 5.91, 6.4, 6.8, 7.05, 8.9, 8.7, 9,9.05]) 

#2. 数据预处理，该案例不需要

#3. 特征方程，该案例不需要

#4. 模型训练
#4.1 分别创建 线性回归 和 回归决策树 模型对象
estimator1 = LinearRegression()
estimator2 = DecisionTreeRegressor(max_depth=1)    #决策树，最大深度为1
estimator3 = DecisionTreeRegressor(max_depth=3)    #决策树，最大深度为3

#4.2 模型训练
estimator1.fit(x_train,y_train)
estimator2.fit(x_train,y_train)
estimator3.fit(x_train,y_train)

#5. 模型预测
#5.1 准备测试集的 特征数据
#x_test = np.array(list(range(0.0,10.0,0.1))).reshape(-1,1) #报错：python支持的range()函数不支持小数
x_test = np.arange(0.0,10.0,0.1).reshape(-1,1)

#5.2 具体的预测动作
y_pre1 = estimator1.predict(x_test)
y_pre2 = estimator2.predict(x_test)
y_pre3 = estimator3.predict(x_test)


#6. 模型评估,此处略

#7. 绘图
#7.1 以真实值（训练集）绘制 散点图
plt.scatter(x_train,y_train,c='gray')
#7.2 以预测值（线性回归，回归决策树）绘制 曲线
plt.plot(x_test,y_pre1,c='red',label= 'LinearRegressin')
plt.plot(x_test,y_pre2,c='blue',label= 'max_depth = 1')
plt.plot(x_test,y_pre3,c='green',label= 'max_depth = 2')
#7.3 显示图例
plt.legend()
#7.4 设置x轴，y轴，标题
plt.xlabel('data')
plt.ylabel('target')


plt.show()

