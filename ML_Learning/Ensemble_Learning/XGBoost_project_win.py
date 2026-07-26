import joblib                                           #保存和加载模型
import numpy as np
import pandas as pd
import xgboost as xgb                                   #极限梯度提升树对象
from collections import Counter                         #统计数据
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold     #分层K折交叉验证，类似于 网格搜索 时cv=折数
from sklearn.utils import class_weight                  #计算样本权重

#1. 定义函数，对 红酒品质源数据  -> 拆分成 训练集 和 测试集 ，并存储到cvs文件中
def dm01_data_split():
    #1. 加载数据
    df = pd.read_csv('E:\\To_learn_Ai\\ML_Learning\\Ensemble_Learning\\data\\红酒品质分类.csv')
    #2. 查看数据集
    #df.info()
    #3. 抽取特征数据 和 标签数据
    x = df.iloc[:,:-1]
    y = df.iloc[:,-1] -3         #最后一列是标签数据,默认范围是[3,8]
    #4. 查看数据
    print(f'查看标签列的分布情况：{Counter(y)}')
    #5. 切分 训练集 和 测试集
    #参1：特征 参2：标签 参3：测试集比例 参4：随机种子 参5：参考数据集的标签分布
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=23,stratify=y)
    #6. 把上述的 训练集特征 和 标签数据拼接在一起，把 测试集特征 和 标签数据拼接在一起。最后写到文件中
    pd.concat([x_train,y_train],axis=1).to_csv(          #axis=1表示横向拼接
        'E:\\To_learn_Ai\\ML_Learning\\Ensemble_Learning\\data\\红酒品质分类_train.csv',index=False)    #忽略索引      
    pd.concat([x_test,y_test],axis=1).to_csv(          #axis=1表示横向拼接
            'E:\\To_learn_Ai\\ML_Learning\\Ensemble_Learning\\data\\红酒品质分类_test.csv',index=False)    #忽略索引      

#2. 定义函数，训练模型，并保存模型
def dm02_trian_model():
    #1. 读取训练集 和 测试集
    train_data = pd.read_csv('.\\ML_Learning\\Ensemble_Learning\\data\\红酒品质分类_train.csv')
    test_data = pd.read_csv('.\\ML_Learning\\Ensemble_Learning\\data\\红酒品质分类_test.csv')
    #2. 提取 训练集 和 测试集 特征数据 和 标签数据
    x_train =  train_data.iloc[:,:-1]
    y_train = train_data.iloc[:,-1] 
    x_test = test_data.iloc[:,:-1]
    y_test = test_data.iloc[:,-1]
    #3. 创建模型对象
    estimator = xgb.XGBClassifier(
        max_depth=5,                #树最大深度
        n_estimators=100,           #树的数量
        learning_rate=0.1,          #学习率
        random_state=23,            #随机种子
        objective='multi:softmax'   #多分类问题
    )
    #加入 平衡权重，因为数据集 样本不均衡
    #参1：平衡权重 参2：标签数据
    sample_weight = class_weight.compute_sample_weight('balanced',y_train)

    #4. 模型训练
    estimator.fit(x_train,y_train,sample_weight=sample_weight)
    #5. 模型评估
    print(f'模型准确率：{estimator.score(x_test,y_test)}')
    #6. 保存模型
    joblib.dump(estimator,'E:\\To_learn_Ai\\ML_Learning\\Ensemble_Learning\\model\\红酒品质分类.pkl')
    print('模型保存成功！')

#3. 定义函数，测试模型
def dm03_use_model():
    #1. 读取训练集 和 测试集
    train_data = pd.read_csv('.\\ML_Learning\\Ensemble_Learning\\data\\红酒品质分类_train.csv')
    test_data = pd.read_csv('.\\ML_Learning\\Ensemble_Learning\\data\\红酒品质分类_test.csv')
    #2. 提取 训练集 和 测试集 特征数据 和 标签数据
    x_train =  train_data.iloc[:,:-1]
    y_train = train_data.iloc[:,-1] 
    x_test = test_data.iloc[:,:-1]
    y_test = test_data.iloc[:,-1]  
    #3. 加载模型
    estimator = joblib.load('E:\\To_learn_Ai\\ML_Learning\\Ensemble_Learning\\model\\红酒品质分类.pkl')
    #4. 创建网格搜索 + 交叉验证（结合分层采样数据），找模型最优参数
    #4.1 定义变量，记录：参数组合
    param_dict = {'max_depth':[2,3,4,5,6,7],
                  'n_estimators':[30,50,100,150],
                  'learning_rate':[0.01,0.2,0.3,1,1.3]}
    #4.2 创建 分层采样 对象
    #参1：折数 参2：是否打乱数据 参3：随机种子
    skf = StratifiedKFold(n_splits=5,shuffle=True,random_state=23)
    #4.3 创建 网格搜索 + 交叉验证（结合分层采样数据）对象
    #参1：模型对象 参2：参数组合 参3：交叉验证对象
    gs_estimator = GridSearchCV(estimator,param_dict,cv=skf)
    #5. 模型训练
    gs_estimator.fit(x_train,y_train)
    #6. 模型预测
    y_pre = gs_estimator.predict(x_test)
    print(f'预测值为“{y_pre}')
    #7. 打印最优评估系数 
    print(f'最优估计器对象组合：{gs_estimator.best_estimator_}')
    print(f'最优评分：{gs_estimator.best_score_}')
#4. 测试
if __name__ == '__main__':
    dm03_use_model()