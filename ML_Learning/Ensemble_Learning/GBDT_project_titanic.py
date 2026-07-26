"""
案例:
演示 Boosting思想之 GBDT(Gradient Boosting Decision Tree, 梯度提升树) 处理 泰坦尼克号数据集。

GBDT 梯度提升树解释:
概述:
    通过拟合 负梯度 来获取一个强学习器

流程:
    1. 采用所有目标值的均值 作为第1个弱学习器的 预测值。
    2. 目标值 - 预测值 = 负梯度(残差), 该(列)值作为 第2个弱学习器的 目标值。
    3. 针对第1个弱学习器, 依次计算每个分割点的 最小平方和, 找到最佳 分割点, 至此: 第1个弱学习器搭建完毕。
    4. 把上述的分割点带入第2个弱学习器, 计算它的预测值 = 以此分割点为界, 目标值的均值, 即为该部分数据的 预测值。
    5. 计算第2个弱学习器的 负梯度, 最佳分割点, 至此: 第2个弱学习器搭建完毕。
    6. 以此类推, 直至程序结束。
"""

#导包
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier             #决策树分类器
from sklearn.ensemble import GradientBoostingClassifier     #梯度提升树分类器
from sklearn.metrics import classification_report,accuracy_score
from sklearn.model_selection import GridSearchCV            #网格搜索

#1. 加载数据
df = pd.read_csv('E:\\To_learn_Ai\\ML_Learning\\Ensemble_Learning\\data\\titanic_train.csv')
#df.info()
#2. 数据的预处理
#2.1 提取 特征 和 标签
x = df[['Pclass','Sex','Age']].copy()
y = df['Survived'].copy()

#2.2 处理Age列的缺失值，用该列均值进行填充
x['Age'] = x['Age'].fillna(x['Age'].mean())

#2.3 热编码处理字符串类型
x = pd.get_dummies(x)

#2.4 切割 训练集 和 测试集
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=23)

#3. 特征工程

#4. 模型训练 预测 评估
#场景1：单个决策树(CART)
#4.1 创建模型对象
estimator1 = DecisionTreeClassifier()
#4.2 模型训练
estimator1.fit(x_train,y_train)
#4.3 模型预测
y_pre1 = estimator1.predict(x_test)
print(f'单个决策树对象的预测结果：{y_pre1}')
#4.4 模型评估
print(f'单个决策树对象的准确率：{accuracy_score(y_test,y_pre1)}')
print('-'*23)

#场景2：梯度提升树(GBDT)
#4.1 创建模型对象
estimator2 = GradientBoostingClassifier()
#4.2 模型训练
estimator2.fit(x_train,y_train)
#4.3 模型预测
y_pre2 = estimator2.predict(x_test)
print(f'单个决策树对象的预测结果：{y_pre2}')
#4.4 模型评估
print(f'单个决策树对象的分类评估报告：{accuracy_score(y_test,y_pre2)}')
print('-'*23)

#场景3：梯度提升树(GBDT)，进行参数优化
#4.1 定义模型可选参数
# GBDT网格搜索参数候选字典
param_dict = {
    # 弱学习器（决策树）数量
    'n_estimators': [ 50, 60, 70, 80, 90, 100,110],
    # 学习率，梯度更新步长
    'learning_rate': [ 0.3, 0.5, 0.6, 0.7],
    # 单棵决策树最大深度，限制树复杂度防止过拟合
    'max_depth': [3, 5, 6, 7, 8, 9]
}
#4.2 创建 梯度提升树 模型对象
estimator3 = GradientBoostingClassifier()
estimator3.fit(x_train,y_train)
#4.3 创建 网格搜索对象
estimator4 = GridSearchCV(estimator3,param_dict,cv=5)
#4.4 模型训练
estimator4.fit(x_train,y_train)
#4.3 模型预测
y_pre4 = estimator4.predict(x_test)
print(f'单个决策树对象的预测结果：{y_pre4}')
#4.4 模型评估
print(f'单个决策树对象的分类评估报告：{accuracy_score(y_test,y_pre4)}')
print('-'*23)