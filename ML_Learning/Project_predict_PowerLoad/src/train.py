# -*- coding: utf-8 -*-
#导包
import os
import sys
import pandas as pd

# # 获取当前文件所在目录 (src)
# current_dir = os.path.dirname(os.path.abspath(__file__))
# # 获取项目根目录 (src 的上一层)
# project_root = os.path.dirname(current_dir)
# # 将根目录加入搜索路径
# sys.path.append(project_root)

import matplotlib.pyplot as plt
import datetime
from utils.log import Logger
from utils.common import data_preprocessing
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error,mean_absolute_error,root_mean_squared_error,mean_absolute_percentage_error
import joblib

plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['font.size'] = 15

#1. 定义电力负荷模型类，配置日志，获取数据源
class PowerLoadModel:
    #1.1 初始化属性信息
    def __init__(self):
        #1.2 拼接日志文件名
        logfile_name = 'train' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        #1.3 创建日志对象
        self.logfile = Logger('../',logfile_name).get_logger()
        #测试写一条日志
        #self.logfile.info('开始创建 电力负荷模型类的 对象了')
        #1.4 获取数据源
        self.data_source = data_preprocessing()

#2. 查看数据的整体分布情况
def ana_data(data):
    #2.1 为了防止修改原数据，做一次拷贝
    ana_data = data.copy()
    #2.2 查看数据的整体分布
    ana_data.info()
    #2.3 负荷整体的分布情况，直方图
    fig = plt.figure(figsize = (20,40))
    ax1 = fig.add_subplot(4,1,1)
    ax1.hist(ana_data['power_load'],bins=100)
    ax1.set_title('负荷整体分布情况')
    ax1.set_xlabel('负荷')

    #2.4 各个小时的平均负荷趋势，看一下负荷在一天中的变化情况
    ana_data['hour'] = ana_data['time'].str[11:13]
    hour_load_mean = ana_data.groupby('hour',as_index=False)['power_load'].mean()
    ax2 = fig.add_subplot(4,1,2)
    ax2.plot(hour_load_mean['hour'],hour_load_mean['power_load'])
    ax2.set_title('每小时负荷平均情况')
    ax2.set_xlabel('小时')

    #2.5 各个月的平均负荷趋势，看一下负荷在一天中的变化情况
    ana_data['month'] = ana_data['time'].str[5:7]
    hour_load_mean = ana_data.groupby('month',as_index=False)['power_load'].mean()
    ax3 = fig.add_subplot(4,1,3)
    ax3.plot(hour_load_mean['month'],hour_load_mean['power_load'])
    ax3.set_title('每月负荷平均情况')
    ax3.set_xlabel('月')

    #2.6 工作日与周末的平均负荷情况，看一下工作日和周末的负荷是否有区别
    ana_data['weekday'] = ana_data['time'].apply(lambda x:pd.to_datetime(x).weekday())
    ana_data['is_holiday'] = ana_data['weekday'].apply(lambda x:1 if x in [5,6] else 0)
    work_load_mean = ana_data[ana_data['is_holiday'] == 0]['power_load'].mean()
    holiday_load_mean = ana_data[ana_data['is_holiday'] == 1]['power_load'].mean()
    ax4 = fig.add_subplot(4,1,4)
    ax4.bar(['工作日','周末'],[work_load_mean,holiday_load_mean])
    ax4.set_title('工作日和周末负荷平均情况')
    plt.show()


#3. 特征工程（重点）
def feature_engineering(data,logger):
    #先拷贝数据
    feature_data = data.copy()
    #1. 提取时间特征，小时，月份
    feature_data['hour'] = feature_data['time'].str[11:13]
    feature_data['month'] = feature_data['time'].str[5:7]
    #热编码 处理hour和month
    hour_month_data = pd.get_dummies(feature_data[['hour','month']])
    feature_data = pd.concat([feature_data,hour_month_data],axis=1)

    #2. 提取相近时间窗口中的负荷特征
    load_1h_data = feature_data['power_load'].shift(1)
    load_2h_data = feature_data['power_load'].shift(2)
    load_3h_data = feature_data['power_load'].shift(3)
    load_shift_data = pd.concat([load_1h_data,load_2h_data,load_3h_data],axis=1)
    load_shift_data.columns = ['前1小时','前2小时','前3小时']
    feature_data = pd.concat([feature_data,load_shift_data],axis=1)

    #3. 提取昨日同时刻负荷特征
    feature_data['yesterday_time'] = feature_data['time'].apply(lambda x:(pd.to_datetime(x) - datetime.timedelta(days=1)).strftime('%Y-%m-%d %H:%M:%S'))
    # 日期和负荷拼接成字典
    time_load_dict = feature_data.set_index('time')['power_load'].to_dict()
    feature_data['yesterday_load'] = feature_data['yesterday_time'].apply(lambda x:time_load_dict.get(x))
    #删除na
    feature_data = feature_data.dropna()

    #5. 整理时间特征，并返回
    feature_columns = list(hour_month_data.columns) + list(load_shift_data.columns) + ['yesterday_load']
    #print(feature_columns)
    #6. 返回结果
    return feature_data,feature_columns

#4. 模型训练
def model_train(data,feature,logger):
    #1. 数据集切分
    x = data[feature]
    y = data['power_load']
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2,random_state = 23)

    #2. 网格化搜索和交叉验证
    logger.info('-------网格搜索+交叉验证 寻找最优超参---------')
    #2.1 定义参数字典
    param_dict = {
        'n_estimators':[50,100,150,200],
        'max_depth':[3,5,6,7],
        'learning_rate':[0.01,0.1]
    }
    #2.2 创建XGBoost 模型对象
    # estimator = XGBRegressor()
    # #2.3 创建网格搜索对象
    # gs = GridSearchCV(estimator, param_grid=param_dict,cv=5)
    # gs.fit(x_train,y_train)
    # logger.info(f'最优参数组合：{gs.best_params_}')
    # logger.info(f'结束时间：{datetime.datetime.now()}')

    #3. 模型实例化
    estimator = XGBRegressor(n_estimators=100,max_depth=7,learning_rate=0.1)
    #4. 模型训练
    estimator.fit(x_train,y_train)
    y_pred = estimator.predict(x_test)
    #5. 模型评估
    print(f'均方误差：{mean_squared_error(y_test, y_pred)}')
    print(f'均方根误差：{root_mean_squared_error(y_test, y_pred)}')
    print(f'平均绝对误差：{mean_absolute_error(y_test, y_pred)}')
    print(f'平均绝对百分比误差：{mean_absolute_percentage_error(y_test, y_pred)}')
    #6. 模型保存
    joblib.dump(estimator, '../model/power_load_model.pkl')

#5. 测试
if __name__ == '__main__':
    #4.1 创建模型对象
    pm  = PowerLoadModel()
    #4.2 打印数据源
    #print(pm.data_source)
    #4.3 查看数据分布
    #ana_data(data=pm.data_source)
    #4.4 特征工程
    feature_data,feature_columns = feature_engineering(pm.data_source,pm.logfile)
    #4.5 模型训练
    #参1：处理后的所有数据 参2：特征名称列表 参3：日志对象
    model_train(feature_data,feature_columns,pm.logfile)