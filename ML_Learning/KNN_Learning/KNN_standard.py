'''
案例：演示特征预处理之 归一化操作。
回顾：特征工程的目的 和 步骤
    目的：利用专业的背景知识 和 技巧处理数据，用于提升 模型的性能。
    步骤：
        特征提取。
        特征预处理 (归一化，标准化)
        特征降维。
        特征选择。
        特征组合。
特征预处理之 归一化介绍：
    目的：
        防止因为量纲 (单位) 问题，导致特征列的方差值相差较大，影响模型的最终结果。
        所以通过公式 把 各列的值 映射到 均值为0 ，标准差为1 的正太分布序列。
    公式：
        x′= （当前值 - 该列平均值）/ σ
    应用场景：
        适用于 大数据集 的处理
'''

#1. 导包
from sklearn.preprocessing import StandardScaler  #标准化对象

#2. 准备数据集（标准化之前的数据）
x_train = [[90,2,10,40],[60,4,15,45],[75,3,13,46]]

#3. 创建标准化对象
#参数 feature_range 表示生成范围,默认为：0,1 如果是这个区间可以省略不写
#transfer = MinMaxScaler(feature_range=(0,1))
transfer = StandardScaler()

#4. 对原数据集进行归一化操作
x_train_new = transfer.fit_transform(x_train)

#5. 打印处理后的数据
print('标准化后的数据\n')
print(x_train_new)

#6. 打印数据集的均值和方差
print(f'数据集的均值：{transfer.mean_}')
print(f'数据集的方差：{transfer.var_}')
print(f'数据集的标准差：{transfer.scale_}')