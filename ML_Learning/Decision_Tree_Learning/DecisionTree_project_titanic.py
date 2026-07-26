#导包
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

#1. 加载数据
data = pd.read_csv('E:\To_learn_Ai\ML_Learning\Decision_Tree_Learning\data\\titanic_train.csv')
#data.info()

#2. 数据预处理
#2.1 提取特征和标签
x = data[['Pclass','Sex','Age']]
y = data['Survived']

#2.2 发现Age列有缺失，我们用该列的 平均值 做填充
# x['Age'].fillna(x['Age'].mean(),inplace=True)       #会警报，但是可以用
# x['Age'] = x['Age'].fillna(x['Age'].mean())         #会警报，因为是直接修改源数据的
#解决方案：copy()数据之后再改
x = x.copy()
x['Age'] = x['Age'].fillna(x['Age'].mean()) 

#2.4 针对Sex列进行热编码
x = pd.get_dummies(x,columns=['Sex'])

#2.5 切分训练集 和测试集
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=23)

#3. 特征工程

#4. 模型训练
#参数：max_depth 绘制的决策树结构 最多10层
estimator = DecisionTreeClassifier(max_depth=10)
estimator.fit(x_train,y_train)

#5. 模型预测
y_pre = estimator.predict(x_test)
print(f'预测值为：{y_pre}')

#6. 模型评估
print(f'分类评估报告：\n{classification_report(y_test,y_pre)}')

#7. 绘制 决策树
plt.figure(figsize=(30,20))
#参1：模型对象 参2：是否用颜色填充 参3：绘制的 决策树结构 ：最多10层
plot_tree(estimator,filled=True,max_depth=10)
plt.savefig('E:\To_learn_Ai\ML_Learning\Decision_Tree_Learning\data\my_titanic.png')
plt.show()
