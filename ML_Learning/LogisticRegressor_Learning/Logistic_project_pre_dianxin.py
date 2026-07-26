import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,classification_report, roc_auc_score


#1. 定义函数，演示：数据的预处理
def dm01_data_preprocess():
    #1. 读取csv文件，获取df对象
    churn_df = pd.read_csv('E:\To_learn_Ai\ML_Learning\LogisticRegressor_Learning\data\churn.csv')
    #2. 查看数据集
    # churn_df.info()
    # print(churn_df.head(5))
    #3. 因为Churn 和 gender 列是字符串，所以需要one-hot编码
    churn_df = pd.get_dummies(churn_df,columns=['Churn','gender'])
    #4. 查看处理后的数据集
    # churn_df.info()
    # print(churn_df.head(5))    
    #5. 删除one-hot处理后，冗余的列
    #参1：要删除的列 参2：1表示删除列 参3：表示直接修改数据
    churn_df.drop(['Churn_No','gender_Male'],axis=1,inplace=True)
    # churn_df.info()
    # print(churn_df.head(5)) 
    #6. 修改列名，将Churn_Yes -> flag，充当标签列
    churn_df.rename(columns={'Churn_Yes':'flag'},inplace=True)
    # churn_df.info()
    # print(churn_df.head(5))  
    #7. 查看数据值的分布
    print(churn_df.flag.value_counts())

#2. 定义函数，演示：查看数据的可视化
def dm02_data_visualization():
    #1. 读取csv文件，获取df对象
    churn_df = pd.read_csv('E:\To_learn_Ai\ML_Learning\LogisticRegressor_Learning\data\churn.csv')    
    #2. 查看数据集
    #3. 因为Churn 和 gender 列是字符串，所以需要one-hot编码
    churn_df = pd.get_dummies(churn_df,columns=['Churn','gender'])
    #4. 查看处理后的数据集
    #5. 删除one-hot处理后，冗余的列
    #参1：要删除的列 参2：1表示删除列 参3：表示直接修改数据
    churn_df.drop(['Churn_No','gender_Male'],axis=1,inplace=True) 
    #6. 修改列名，将Churn_Yes -> flag，充当标签列
    churn_df.rename(columns={'Churn_Yes':'flag'},inplace=True)  
    #7. 查看列名，方便特征提取
    print(churn_df.columns)
    '''
    Index(['Partner_att', 'Dependents_att', 'landline', 'internet_att',
       'internet_other', 'StreamingTV', 'StreamingMovies', 'Contract_Month',
       'Contract_1YR', 'PaymentBank', 'PaymentCreditcard', 'PaymentElectronic',
       'MonthlyCharges', 'TotalCharges', 'flag', 'gender_Female'],
      dtype='object')
    '''
    #8. 数据的可视化,绘制 计数柱状图
    #参1：数据集 参2：x轴的列名（月度会员）参3：hue表示分组，根据分组进行绘制，这里是：是否流失（false->不流失）
    sns.countplot(data=churn_df,x='Contract_Month',hue='flag')
    plt.show()

#3. 定义函数，演示：逻辑回归算法的模型训练，预测，评估
def dm03_logistic_regression():
    #1. 读取csv文件，获取df对象
    churn_df = pd.read_csv('E:\To_learn_Ai\ML_Learning\LogisticRegressor_Learning\data\churn.csv')    
    
    #2. 数据预处理
    #2.1 因为Churn 和 gender 列是字符串，所以需要one-hot编码
    churn_df = pd.get_dummies(churn_df,columns=['Churn','gender'])
    #2.2 删除one-hot处理后，冗余的列
    #参1：要删除的列 参2：1表示删除列 参3：表示直接修改数据
    churn_df.drop(['Churn_No','gender_Male'],axis=1,inplace=True) 
    #2.3 修改列名，将Churn_Yes -> flag，充当标签列
    churn_df.rename(columns={'Churn_Yes':'flag'},inplace=True)  
    #2.4 提取特征列 和 标签列
    x = churn_df[['Contract_Month','internet_other','PaymentElectronic']]
    '''
    x = churn_df[['Partner_att', 'Dependents_att', 'landline', 'internet_att',
       'internet_other', 'StreamingTV', 'StreamingMovies', 'Contract_Month',
       'Contract_1YR', 'PaymentBank', 'PaymentCreditcard', 'PaymentElectronic',
       'MonthlyCharges', 'TotalCharges', 'gender_Female']]
    '''
    y = churn_df['flag']
    #2.5 划分 训练集  和 测试集
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=23)

    #3. 特征工程（特征提取，预处理）

    #4. 模型训练
    #4.1 创建训练对象
    estimator = LogisticRegression()
    #4.2 模型训练
    estimator.fit(x_train,y_train)

    #5. 模型预测
    y_pre = estimator.predict(x_test)
    print(f'预测值：{y_pre}')

    #6. 模型评估
    print(f'准确率：{accuracy_score(y_test,y_pre)}')

    print(f'精确率：{precision_score(y_test,y_pre)}')
    print(f'召回率：{recall_score(y_test,y_pre)}')    
    print(f'F1：{f1_score(y_test,y_pre)}')

    print(f'分类评估报告：\n{classification_report(y_test,y_pre)}')

if __name__ == '__main__':
    dm03_logistic_regression()
