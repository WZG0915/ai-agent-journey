import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import joblib                       #保存模型
from collections import Counter     #计数统计

#1. 定义函数，接收用户传入的索引，展示 索引对应的图片
def show_digit(idx):
    #1.读取数据集,h获取源数据
    df = pd.read_csv('data/手写数字识别.csv')
    #print(df) 
    #2.判断传入的索引是否越界
    if idx < 0 or idx >  len(df)-1 :
        print('索引越界！')
    #3.走这里，说明没有越界，正常获取数据
    x = df.iloc[:,1:]
    y = df.iloc[:,0]
    #4.查看用户传入的索引对应的图片是几？
    #print(f'该图片对应的数字是：{y.iloc[idx]}')
    #5.查看 x 的形状
    print(x.iloc[idx].shape)
    print(x.iloc[idx].values)
    #6.把(784,)转化为(28,28)
    x = x.iloc[idx].values.reshape(28,28)
    #7.绘制灰度图的动作
    plt.imshow(x,cmap='gray')   #灰度图
    plt.axis('off')
    plt.show()

#2. 定义函数，训练模型，并保存训练好的模型
def train_model():
    #1.加载数据集
    df = pd.read_csv('ML_Learning/KNN_Learning/data/手写数字识别.csv')

    #2.数据预处理
    #2.1 拆分特征列
    x = df.iloc[:,1:]   #特征列
    #2.2 拆分标签列
    y = df.iloc[:,0]    #标签列
    #2.3 打印标签形状
    #print(f'x的形状：{x.shape}')
    #print(f'y的形状：{y.shape}')
    #2.4 对特征列（拆分前）进行 归一化
    x = x/255
    #2.5 拆分训练集和测试集
    #参1：特征列 参2：标签列 参3：测试集的比例 参4：随机种子 参5：参考y值进行抽取，保持标签的比例（数据均衡）
    x_train,x_test,y_train,y_test = train_test_split(
        x,y,test_size=0.2,random_state=25,stratify=y
    )

    #3.模型训练
    #3.1 创建模型对象
    estimator = KNeighborsClassifier(n_neighbors=3)
    #3.2 模型训练
    estimator.fit(x_train,y_train)

    #4. 模型评估
    print(f'准确率：{estimator.score(x_test,y_test)}')
    print(f'准确率：{accuracy_score(y_test,estimator.predict(x_test ))}')    

    #5. 保存模型
    #参1：模型对象 参2：模型保存的路径
    joblib.dump(estimator,filename='my_model/Num_Recognize.pkl')    #pickle文件：python（Pandas）独有的文件类型 
    print('模型保存成功！')

#3. 定义函数，测试模型
def use_model():
    #1. 加载图片
    x = plt.imread('data/demo.png') #28*28
    #2. 绘制图片
    # plt.imshow(x,cmap='gray')   #灰度图
    # plt.axis('off')             #不显示坐标
    # plt.show()
    #3. 加载模型
    estimator = joblib.load('my_model/Num_Recognize.pkl')
    #4. 模型预测
    #4.1 查看 数据集转换
    # print(x.shape)
    # print(x.reshape(1,-1).shape)      #语法糖
    #4.2 具体的转换动作，记得归一化
    x = x.reshape(1,-1)
    #4.3 模型预测
    y_pre = estimator.predict(x)

    #5. 打印预测结果
    print(f'预测结果为：{y_pre}')


#4，测试
if __name__ == '__main__':
    train_model()
    use_model()
