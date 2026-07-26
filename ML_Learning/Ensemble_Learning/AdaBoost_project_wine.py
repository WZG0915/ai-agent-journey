#导包
import pandas as pd
from sklearn.preprocessing import LabelEncoder          #标签编码器
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score

#1. 加载数据
df_wine = pd.read_csv('ML_Learning\\Ensemble_Learning\\data\\wine0501.csv')
#df_wine.info()
#print(df_wine['Class label'].unique())      #葡萄酒类别有3种，但是决策树只能识别 二叉树

#2. 数据预处理
#2.1 从 标签列(Class label)中，过滤掉 1类别，剩下 2 3类别
df_wine = df_wine[df_wine['Class label'] != 1]
#print(df_wine['Class label'].unique())
#2.2 获取 特征列 和 标签列
x = df_wine[['Alcohol','Hue']]
y = df_wine[['Class label']]
#2.4 通过 标签编码器 把 标签列 转换为 数值列
le = LabelEncoder()
y = le.fit_transform(y)                 #[2,3] -> [0,1]
#2.5 训练集 和 测试集 分割
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=23,stratify=y)

#3. 特征工程

#4. 模型训练，预测，评估
#场景1：单一决策树 -> 充当弱分类器
#4.1 创建模型对象
estimator1 = DecisionTreeClassifier(max_depth=3)
#4.2 训练模型
estimator1.fit(x_train,y_train)
#4.3 模型预测
y_pre1 = estimator1.predict(x_test)
#4.4 模型评估
print(f'单一决策树准确率：{accuracy_score(y_test,y_pre1)}')

#4. 模型训练，预测，评估
#场景1：AdaBoost模型 -> 集成学习，CART树，200棵
#4.1 创建模型对象
#参1：弱分类器（决策树对象） 参2：弱分类器个数 参3：学习率
estimator2 = AdaBoostClassifier(estimator=estimator1,n_estimators=200,learning_rate=0.1)
#4.2 训练模型
estimator2.fit(x_train,y_train)
#4.3 模型预测
y_pre2 = estimator2.predict(x_test)
#4.4 模型评估
print(f'AdaBoost模型准确率：{accuracy_score(y_test,y_pre2)}')
