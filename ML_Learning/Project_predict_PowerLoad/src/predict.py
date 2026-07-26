# -*- coding: utf-8 -*-
import sys
import os

# 1. 动态将项目根目录追加至 sys.path，防止出现 ModuleNotFoundError 报错
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pandas as pd
import numpy as np
import datetime

from utils.log import Logger
from utils.common import data_preprocessing
from sklearn.metrics import mean_absolute_error
import matplotlib.ticker as mick
import joblib
import matplotlib.pyplot as plt

# 设置画图字体，解决中文与负号显示问题
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 15


# 1. 预测特征提取（严格与训练时的 40 个特征对齐）
def pred_feature_extract(data_dict, time, logger):
    """
    预测数据解析特征，保持与模型训练时的特征列名一致
    1.解析时间特征 (24个小时特征 + 12个月份特征)
    2.解析时间窗口特征 (前1小时, 前2小时, 前3小时)
    3.解析昨日同时刻特征 (yesterday_load)
    """
    logger.info(f'=========解析预测时间为：{time}所对应的特征==============')
    # 特征列清单（共 40 列）
    feature_names = ['hour_00', 'hour_01', 'hour_02', 'hour_03', 'hour_04', 'hour_05',
                     'hour_06', 'hour_07', 'hour_08', 'hour_09', 'hour_10', 'hour_11',
                     'hour_12', 'hour_13', 'hour_14', 'hour_15', 'hour_16', 'hour_17',
                     'hour_18', 'hour_19', 'hour_20', 'hour_21', 'hour_22', 'hour_23',
                     'month_01', 'month_02', 'month_03', 'month_04', 'month_05', 'month_06',
                     'month_07', 'month_08', 'month_09', 'month_10', 'month_11', 'month_12',
                     '前1小时', '前2小时', '前3小时', 'yesterday_load']

    # 1.1 小时特征数据（24 维）
    hour_part = []
    pred_hour = time[11:13]
    for i in range(24):
        if pred_hour == feature_names[i][5:7]:
            hour_part.append(1)
        else:
            hour_part.append(0)

    # 1.2 月份特征数据（12 维）-- 【已修复：修正判断变量为 pred_month，并正确追加至 month_part】
    month_part = []
    pred_month = time[5:7]
    for i in range(24, 36):
        if pred_month == feature_names[i][6:8]:
            month_part.append(1)
        else:
            month_part.append(0)

    # 2. 历史窗口及昨日负荷特征（4 维）
    # 前1小时负荷
    last_1h_time = (pd.to_datetime(time) - pd.to_timedelta('1h')).strftime('%Y-%m-%d %H:%M:%S')
    last_1h_load = data_dict.get(last_1h_time, 600)

    # 前2小时负荷
    last_2h_time = (pd.to_datetime(time) - pd.to_timedelta('2h')).strftime('%Y-%m-%d %H:%M:%S')
    last_2h_load = data_dict.get(last_2h_time, 600)

    # 前3小时负荷
    last_3h_time = (pd.to_datetime(time) - pd.to_timedelta('3h')).strftime('%Y-%m-%d %H:%M:%S')
    last_3h_load = data_dict.get(last_3h_time, 600)

    # 昨日同时刻负荷 -- 【已修复：传入计算出的时间字符串 last_day_time 进行查询】
    last_day_time = (pd.to_datetime(time) - pd.to_timedelta('1d')).strftime('%Y-%m-%d %H:%M:%S')
    last_day_load = data_dict.get(last_day_time, 600)

    # 汇总历史特征
    his_part = [last_1h_load, last_2h_load, last_3h_load, last_day_load]

    # 特征数据拼接 (24 + 12 + 4 = 40)
    feature_list = [hour_part + month_part + his_part]
    feature_df = pd.DataFrame(feature_list, columns=feature_names)

    # 【已修复：同时返回特征 DataFrame 和列名清单，保证列顺序对齐】
    return feature_df, feature_names


# 2. 绘图函数
def prediction_plot(data):
    """
    绘制时间与预测负荷折线图，时间与真实负荷折线图，展示预测效果
    :param data: 数据一共有三列：时间、真实值、预测值
    """
    fig = plt.figure(figsize=(40, 20))
    ax = fig.add_subplot()

    # 绘制时间与真实负荷的折线图
    ax.plot(data['时间'], data['真实值'], label='真实值', color='blue')
    # 绘制时间与预测负荷的折线图 -- 【已修复：画图列名改为 预测值，颜色改为 红色】
    ax.plot(data['时间'], data['预测值'], label='预测值', color='red')

    ax.set_ylabel('负荷')
    ax.set_xlabel('时间')
    ax.set_title('预测负荷以及真实负荷的折线图')

    # 调整横坐标展示间隔与角度
    ax.xaxis.set_major_locator(mick.MultipleLocator(50))
    plt.xticks(rotation=45)
    plt.legend()

    plt.savefig('../data/fig/预测效果.png')
    plt.show()


# 3. 配置电力负荷预测类
class PowerLoadPredict(object):
    def __init__(self, filename):
        # 配置日志记录
        logfile_name = "predict_" + datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        self.logfile = Logger('../', logfile_name).get_logger()
        # 获取数据源
        self.data_source = data_preprocessing(filename)
        # 历史数据转为字典 key:时间，value:负荷
        self.data_dict = self.data_source.set_index('time')['power_load'].to_dict()


# 4. 主流程测试
if __name__ == '__main__':
    # 4.1 实例化预测类
    input_file = os.path.join('../data', 'test.csv')
    pred_obj = PowerLoadPredict(input_file)

    # 4.2 加载训练好的模型 (根据你实际保存的模型文件名修改，如 xgb.pkl 或 power_load_model.pkl)
    model_path = '../model/xgb.pkl' if os.path.exists('../model/xgb.pkl') else '../model/power_load_model.pkl'
    model = joblib.load(model_path)

    # 4.3 确定要预测的时间段：2015-08-01 00:00:00 及以后的时间
    evaluate_list = []
    pred_times = pred_obj.data_source[pred_obj.data_source['time'] >= '2015-08-01 00:00:00']['time']

    # 4.4 滚动模拟真实场景进行负荷预测
    for pred_time in pred_times:
        print(f"开始预测时间为：{pred_time}的负荷")
        # 【已修复：将 pp.logger() 改为标准的 pred_obj.logfile.info()】
        pred_obj.logfile.info(f"开始预测时间为：{pred_time}的负荷")

        # 【已修复：采用动态掩码 k < pred_time，使模型能持续获取最新历史数据】
        data_his_dict = {k: v for k, v in pred_obj.data_dict.items() if k < pred_time}

        # 4.5 解析特征并预测
        processed_data, feature_cols = pred_feature_extract(data_his_dict, pred_time, pred_obj.logfile)

        # 4.6 显式切片保证特征列名顺序完全一致
        pred_value = model.predict(processed_data[feature_cols])

        # 4.7 收集真实值与预测值
        true_value = pred_obj.data_dict.get(pred_time)
        pred_obj.logfile.info(f"真实负荷为：{true_value}, 预测负荷为：{pred_value[0]}")
        evaluate_list.append([pred_time, true_value, pred_value[0]])

    # 4.8 转换为 DataFrame 对象
    evaluate_df = pd.DataFrame(evaluate_list, columns=['时间', '真实值', '预测值'])

    # 5. 预测结果评价
    # 【已修复：修正字符串引号冲突】
    mae_score = mean_absolute_error(evaluate_df['真实值'], evaluate_df['预测值'])
    print(f"模型对新数据进行预测的平均绝对误差：{mae_score}")
    pred_obj.logfile.info(f"模型对新数据进行预测的平均绝对误差：{mae_score}")

    # 5.2 绘制并保存折线图
    prediction_plot(evaluate_df)