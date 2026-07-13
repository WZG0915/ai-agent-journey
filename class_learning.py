#类定义
class people:
    #定义基本属性
    # name = '' 
    # age = 0
    # #私有属性，无法在类外部访问
    # __weight = 0
    #定义构造方法
    def __init__(self, n ,a ,w):
        self.name = n
        self.age = a
        self.__weight = w
    def speak(self):
        print(f'{self.name}说他的年龄是{self.age}')

#实例化
p1 = people("Bob",20,60)
p1.speak()

#单继承示例
class student(people):
    grade = ''
    def __init__(self, n, a, w,g):
        #调用父类的构函
        people.__init__(self,n ,a ,w)
        self.grade = g
    #覆写父类的方法
    def speak(self):
        print(f'{self.name}说他的年龄是{self.age},他的年级是{self.grade}')

#实例化2
p2 = student("Mike",20,60,6)
p2.speak()