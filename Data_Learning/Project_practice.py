# 1. 导入库
import  numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
rcParams['font.sans-serif'] = ['SimHei']  #设置字体

# 2. 导入数据
df = pd.read_csv('Data_Learning/data/house_sales.csv')



# 3. 数据概览
#print("总记录数:",len(df))
#print("字段数量:",len(df.columns))
#print(df.info())
df.head(5)
# 4. 数据清洗
#删除无用的数据列
df.drop(columns='origin_url',inplace=True)

#检查是否有缺失值
#print(df.isna().sum())
#删除缺失值
df.dropna(inplace=True)
#print(len(df))

#检测是否有重复数据
#print(df.duplicated().sum())
#删除重复数据
df.drop_duplicates(inplace=True)
#print(df.head())

#数据类型转换
#面积的数据类型转换
df['area'] = df['area'].str.replace('㎡','').astype(float)
#售价的数据类型转换
df['price'] = df['price'].str.replace('万','').astype(float)
#朝向的数据类型转换
#print(df['toward'].value_counts())
df['toward'] = df['toward'].astype('category')
#单价的数据类型转换
df['unit'] = df['unit'].str.replace('元/㎡','').astype(float)
#建造年份的数据类型转换
df['year'] = df['year'].str.replace('年建','').astype(int)

#异常值的处理
#房屋面积的异常处理
#print(df[df['area']<20]) 
df = df[(df['area']<600) & (df['area']>20)]
#房屋售价的异常处理     IQR
Q1 = df['price'].quantile(0.25)
Q3 = df['price'].quantile(0.75)
IQR = Q3 - Q1
low_price = Q1 - 1.5*IQR
high_price = Q3 + 1.5*IQR
df = df[(df['price']< high_price) & (df['price']> low_price)]
# 5. 新数据特征构造
#地区district
df['district'] = df['address'].str.split('-').str[0] 
#楼层的类型floor_type
#df['floor_type'] = df['floor'].str.split('（').str[0]
def fun1(str1):
    if pd.isna(str1):
        return '未知'
    elif '低' in str1:
        return '低楼层'
    elif '中' in str1:
        return '中楼层'
    elif '高' in str1:
        return '高楼层'
    else:
        return '未知'
df['floor_type2']= df['floor'].apply(fun1).astype('category')
#是否是直辖市zxs
def fun2(str2):
    if str2 in ['北京','上海','天津','重庆']:
        return True
    else:
        return False
df['zxs'] = df['city'].apply(lambda x: fun2(x))
#卧室的数量bedrooms
df['bedrooms'] = df['rooms'].str.split('室').str[0].astype(int)
#客厅的数量livingrooms
#df['livingrooms'] = df['rooms'].str.split('室').str[1].str.split('厅').str[0].astype(int)
df['livingrooms'] = df['rooms'].str.extract(r'(\d+)厅').astype(int)
#楼龄building_age
df['building_age'] = 2025 -df['year']
#价格的分段price_labels
df['price_labels'] = pd.cut(df['price'],bins=4,labels=['低价','中价','高价','豪华'])

# 6. 问题分析及可视化
'''
问题编号: A1
问题: 哪些变量最影响房价？面积、楼层、房间数哪个影响更大？
分析主题: 特征相关性
分析目标: 了解房屋各特征对房价的线性影响
分组字段: 无
指标/方法: 皮尔逊相关系数
'''
#选择数值型特征
a = df[['price','area','unit','building_age']].corr()   #相关系数
#对房价的影响最大的几个因素的排序
a['price'].sort_values(ascending=False)[1:]
#相关性的热力图
plt.figure(figsize= (10,10))   
sns.heatmap(a,cmap= 'coolwarm') #cmap风格
plt.title('房屋特征相关性热力图')
plt.tight_layout()
plt.show()


'''
问题编号: A2
问题: 全国房价总体分布是怎样的？是否存在极端值？
分析主题: 描述性统计
分析目标: 概览数值型字段的分布特征
分组字段: 无
指标/方法: 平均数/中位数/四分位数/标准差
'''
print(df.describe())
#房价分布直方图
plt.subplot(111)
#plt.hist(df['price'],bins=10)
sns.histplot(data=df,x='price')
plt.show()

'''
问题编号: A6
问题: 南北向是否真比单一朝向贵？贵多少？
分析主题: 朝向溢价
分析目标: 评估不同朝向的价格差异
分组字段: toward
指标/方法: 方差分析/多重比较
'''
df.groupby('toward').agg({
    'price':['mean','median'],
    'unit':'median',
    'building_age':'mean',
}
)   #聚类函数

#数据可视化
plt.figure(figsize=(14,5))
sns.boxplot(x='toward',y='price',data=df)
plt.tight_layout()
plt.show()