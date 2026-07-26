"""
案例：通过 KNN 算法实现 鸢尾花的 分类操作。

回顾：机器学习项目的研发流程

    1. 加载数据。
    2. 数据的预处理。
    3. 特征工程 (提取，预处理...)
    4. 模型训练。
    5. 模型评估。
    6. 模型预测。
"""

#导入工具包
from sklearn.datasets import load_iris      #加载鸢尾花测试集
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split    #分割训练集和测试集
from sklearn.preprocessing import StandardScaler        #数据标准化
from sklearn.neighbors import KNeighborsClassifier      #KNN算法 分类对象
from sklearn.metrics import accuracy_score              #模型评估的 计算模型预测的准确率

#1. 定义函数，加载鸢尾花数据集，并查看数据集
def dm01_loadiris():
    #1. 加载鸢尾花数据集：
    iris_data = load_iris()
    #2. 查看数据集
    #print(f'数据集：{iris_data}')               #字典形态
    #print(f'数据集的类型：{type(iris_data)}')   
    #3. 查看数据集的所有键
    print(f'数据集的所有键：{iris_data.key()}')
    #4. 查看数据集的键对应的值
    print(f'具体的数据：{iris_data.data[:5]}')      #有150条数据 每条数据有4个特征 我们只看前5条
    print(f'具体的标签：{iris_data.target[:5]}')    #有150条数据，每条数据有1个标签，只看前5条
    print(f'标签对应的名称：{iris_data.target_names}')  #['setosa' 'versicolor' 'virginica']
    print(f'特征对应的名称：{iris_data.feature_names}') 
    #   ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

#2. 定义函数，绘制函数的散点图
def dm02_showiris():
    #1. 加载数据集
    iris_data = load_iris()
    #2. 把鸢尾花数据集封装成 DataFrame对象
    iris_df = pd.DataFrame(iris_data.data, columns=iris_data.feature_names)
    print(iris_df)
    #3. 给df对象新增一列 -> 标签列
    iris_df['label'] = iris_data.target
    #4. 通过 Seaborn 绘制散点图
    #参1：数据集 参2：x轴 参3：y轴 参4：分组字段 参5：是否显示拟合线
    sns.lmplot(data=iris_df,x='sepal length (cm)',y='sepal width (cm)',hue='label',fit_reg=True)
    #5. 设置标题，显示
    plt.title('iris data')
    plt.tight_layout()      #自动调整子图参数 
    plt.show()

#3. 定义函数，切分训练集和测试集
def dm03_split_train_test():
    #1. 加载数据集
    iris_data = load_iris()

    #2. 数据的预处理：从150个特征和标签中 按照8:2的比例，切分训练集和测试集
    # 参1：特征数据 参2：标签数据 参3：测试集的比例 参4：随机种子
    #返回值：训练集的特征数据，测试集的特征数据，训练集的标签数据，测试集的标签数据
    x_train,x_test,y_train,y_test = train_test_split(iris_data.data,
                                                     iris_data.target,
                                                     test_size=0.2,
                                                     random_state=23)
    #3. 打印切割后的数据

#4. 定义函数，实验鸢尾花完整案例 -> 加载数据，数据预处理，特征工程，模型训练，模型评估，模型预测
def dm04_iris_evaluate_test():
    #1. 加载数据集
    iris_data = load_iris()
    
    #2. 数据预处理
    x_train,x_test,y_train,y_test = train_test_split(iris_data.data,
                                                     iris_data.target,
                                                     test_size=0.2,
                                                     random_state=23)
    
    #3. 特征工程（提取，预处理）
    #3.1 创建标准化数据
    transfer = StandardScaler()
    #3.2 对特征列进行标准化
    #fit_transform:兼具fit和transform的功能，即：训练，转换。 该函数适用于：第一次进行标准化的时候使用
    #一般用于处理：训练集
    x_train = transfer.fit_transform(x_train)
    #transform:只有转换，该函数适用于：重复进行标准化动作时使用
    #一般用于对测试集进行标准化
    x_test = transfer.transform(x_test)
    
    #4. 模型训练
    #4.1 创建模型对象
    estimator = KNeighborsClassifier(n_neighbors=3)
    #4.2 具体的训练模型的动作
    estimator.fit(x_train,y_train)   #传入：训练集的特征数据，训练集的标签数据 

    #5. 模型预测
    #场景1：对刚才切分的 测试集（30条）进行预测
    #5.1 直接预测即可，获取预测值
    y_pre = estimator.predict(x_test)
    #5.2 打印预测结果
    print(f'预测值为：{y_pre}')

    #场景2：对新的数据集（源数据150条之外的数据）进行测试
    #5.1 自定义测试数据集
    my_data = [[7.8,2.1,3.9,1.6]]
    #5.2 对数据集进行标准化处理
    my_data = transfer.transform(my_data)
    #5.3 模型评估预测
    y_pre_new = estimator.predict(my_data)
    print(f'最新数据预测：{y_pre_new}')
    #5.4 查看上述数据集每种分类的预测概率
    y_pre_proba= estimator.predict_proba(my_data)
    print(f'各类的预测概率：{y_pre_proba}')    

    #6. 模型评估
    #方式1：直接评分，基于训练集的特征和训练集的标签
    print(f'正确率（准确率）：{estimator.score(x_train,y_train)}')

    #方式2：基于 测试集的标签 和 测试集的预测结果 进行评估
    #主要用方法2
    print(f'正确率（准确率）：{accuracy_score(y_test,y_pre)}')
#5. 测试
if __name__ == '__main__':
    dm04_iris_evaluate_test()