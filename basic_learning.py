import os
import datetime

current_datetime = datetime.datetime.now()
print(current_datetime)

directory = os.getcwd()
print(directory)

files = os.listdir(directory)
print(files)

dir(os)
help(os)

#1. 字符串
var1 = "Hello World"
var2 = 'WZG 0915'
print(var1)
print(var2)
print('var2[1:5]')
#f''的用法
var3 = f'information'
var4 = "Wang Zhaoguo"
print(f'{var4} is hard 1 + 1 {1+1}\n')

#2. 列表
list1 = ["num1", "num2",19,20]
print(list1)
list1[2] = "num3"
print(list1)
list1.append(21)
print(list1)
del list1[2]
print(f"{list1}\n")

#3. 元组
tup1 = (0,1,"love")
print(tup1)
#元组和列表的重要区别 ： 元组不能够修改元素，列表可以修改元素
print()

#4. 字典
#字典中的键不可以重复
#eg dict = {key1:value1 ,key2:value2, key3:value3}
dict1 = {"name": "Wang Zhaoguo", "birthday": "9.15", "place": "Huaibei"}
print(dict1)
print(dict1["name"])
print(f'这个人叫{dict1["name"]}')
del dict1["place"]
print(dict1)
print()

#5. 集合
#集合中的元素不可以重复
set1 = {"num1",1,2}
print(set1)
set2 = set([1,2,tup1])
print(set2)
set3 = set('wang zhao guo')
print(set3)
set2.add("lizhi")
print(set2)
set2.remove("lizhi")
print(set2)
#python推导式
names = ['Bob','Tom','alice','Jerry','Wendy','Smith']
new_names = [name.upper()for name in names if len(name)>3]
print(new_names)