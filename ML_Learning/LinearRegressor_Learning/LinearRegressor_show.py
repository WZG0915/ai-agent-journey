"""
线性回归介绍(Linear Regressor)：
概述/目的：
    用线性公式 来描述 多个自变量(特征) 和 1个因变量(标签)之间 关系的，对其关系进行建模，基于特征 预测 标签。
    线性回归属于：有监督学习，即：有特征，有标签，且标签是连续的。
分类：
    一元线性回归： 1个特征列 + 1个标签列
    多元线性回归： 多个特征列 + 1个标签列
公式：
    一元线性回归：
        y = kx + b ⇒ wx + b
            k：数学中叫斜率，在机器学习中Weight(权重)，简称：w
            b：数学中叫截距，在机器学习中Bias(偏置)，简称：b
    多元线性回归：
        y = w1x1 + w2x2 + w3x3 + ... + wnxn + b
          = w的转置 \* x + b
"""

#导包
from sklearn.linear_model import LinearRegression

#案例：演示线性回归API入门

#1.准备数据
x_train = [[160],[166],[172],[174],[180]]
y_train = [56.3,60.6,65.1,68.5,75]
x_test = [[176]]
 
#2.数据预处理，这里不需要
#3.特征工程（特征提取，特征预处理）

#4.模型训练
#4.1创建模型对象
estimator = LinearRegression()
#4.2具体的训练动作
estimator.fit(x_train,y_train)
#4.3因为是线性回归模型，我们可以查看：斜率（w,权重），截距（b,偏置）
print(f'权重：{estimator.coef_}')
print(f'偏置：{estimator.intercept_}')

#5.模型预测
y_pre = estimator.predict(x_test)
print(f'预测值：{y_pre}')