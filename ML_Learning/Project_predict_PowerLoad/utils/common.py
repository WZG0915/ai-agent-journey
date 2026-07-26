#导包
import  numpy as np
import pandas as pd


#该工具的目的是：对数据进行预处理 -> 时间格式化，按照时间升序排列，对数据进行去重
#数据集在 data目录下 train.csv文件中 -> 拆分 训练集 和 测试集
#测试集在 data目录下 test.csv文件中 -> 模拟项目上线后，真实的测试集

#定义函数，对数据进行预处理操作
def data_preprocessing(file_path):
    #1. 加载数据
    data = pd.read_csv(file_path)

    #2. 时间格式化，转为：'%Y-%m-%d-%H:%M:%S'
    data['time'] = pd.to_datetime(data['time']).dt.strftime('%Y-%m-%d %H:%M:%S')

    #3. 按照时间升序排列
    data.sort_values('time',ascending=True, inplace=True)

    #4. 去重
    data.drop_duplicates(inplace=True)

    #5. 打印和返回
    return data

if __name__ == '__main__':
    data_preprocessing(file_path)