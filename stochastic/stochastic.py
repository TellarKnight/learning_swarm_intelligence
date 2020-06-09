import numpy as np
import matplotlib.pyplot as plt
import time
import random as rdm

# 建立“蚂蚁”类
class Ant(object):
    def __init__(self, path):
        self.path = path  # 蚂蚁当前迭代整体路径
        self.length = self.calc_length(path)  # 蚂蚁当前迭代整体路径长度

    def calc_length(self, path_):  # path=[A, B, C, D, A]注意路径闭环
        length_ = 0
        for i in range(len(path_) - 1):
            delta = (path_[i].x - path_[i + 1].x, path_[i].y - path_[i + 1].y)
            length_ += np.linalg.norm(delta)
        return length_

    @staticmethod
    def calc_len(A, B):  # 静态方法，计算城市A与城市B之间的距离
        return np.linalg.norm((A.x - B.x, A.y - B.y))


# 建立“城市”类
class City(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y


# 建立“路径”类
class Path(object):
    def __init__(self, A):  # A为起始城市
        self.path = [A, A]

    def add_path(self, B):  # 追加路径信息，方便计算整体路径长度
        self.path.append(B)
        self.path[-1], self.path[-2] = self.path[-2], self.path[-1]


# 构建“蚁群算法”的主体
class ACO(object):
    def __init__(self, ant_num=50, maxIter=300):
        self.ants_num = ant_num  # 蚂蚁个数
        self.maxIter = maxIter  # 蚁群最大迭代次数
        ###########################
        # self.deal_data('coordinates-of-cities.dat')  # 提取所有城市的坐标信息
        self.deal_data('eil51_tsp.txt')  # 提取所有城市的坐标信息
        ###########################
        self.path_seed = np.zeros(self.ants_num).astype(int)  # 记录一次迭代过程中每个蚂蚁的初始城市下标
        self.ants_info = np.zeros((self.maxIter, self.ants_num))  # 记录每次迭代后所有蚂蚁的路径长度信息
        self.best_path = np.zeros(self.maxIter)  # 记录每次迭代后整个蚁群的“历史”最短路径长度
        ###########################
        self.solve()  # 完成算法的迭代更新
        self.display()  # 数据可视化展示

    def deal_data(self, filename):
        with open(filename, 'rt') as f:
            temp_list = list(line.split() for line in f)  # 临时存储提取出来的坐标信息
        self.cities_num = len(temp_list)  # 1. 获取城市个数
        self.cities = list(City(float(item[0]), float(item[1])) for item in temp_list)  # 2. 构建城市列表
        self.best_cities_num = np.zeros(self.cities_num + 1)  # 初始化最佳路径城市下标



    def solve(self):
        iterNum = 0  # 当前迭代次数
        while iterNum < self.maxIter:
            self.random_seed()  # 使整个蚁群产生随机的起始点
            ##########################################################################
            for i in range(self.ants_num):
                city_index1 = self.path_seed[i]  # 每只蚂蚁访问的第一个城市下标
                ant_path = Path(self.cities[city_index1])  # 记录每只蚂蚁访问过的城市
                tabu = [city_index1]  # 记录每只蚂蚁访问过的城市下标，禁忌城市下标列表
                non_tabu = list(set(range(self.cities_num)) - set(tabu))
                for j in range(self.cities_num - 1):  # 对余下的城市进行访问
                  for k in range(150):
                    city_index2=rdm.randint(0,50)
                    if city_index2 not in tabu:
                        ant_path.add_path(self.cities[city_index2])
                        tabu.append(city_index2)
                        non_tabu = list(set(range(self.cities_num)) - set(tabu))
                        break
                  city_index1 = city_index2
                self.ants_info[iterNum][i] = Ant(ant_path.path).length
                if iterNum == 0 and i == 0:  # 完成对最佳路径城市的记录
                    self.best_cities = ant_path.path
                else:
                    if self.ants_info[iterNum][i] < Ant(self.best_cities).length: self.best_cities = ant_path.path
                tabu.append(tabu[0])  # 每次迭代完成后，使禁忌城市下标列表形成完整闭环
            self.best_path[iterNum] = Ant(self.best_cities).length
            for u in range(0,self.cities_num):
              for z in range(0,self.cities_num):
                if self.cities[z] == self.best_cities[u]:
                    self.best_cities_num[u]=z
                    break
            self.best_cities_num[self.cities_num]=self.best_cities_num[0]
            iterNum += 1
        for i in range(self.maxIter):
            if(self.best_path[i] == self.best_path[self.maxIter-1]):
                print('算法最少使用%d次迭代得到最优解'%(i+1))
                break
        print('达到最大迭代次数后得到的最短路径长度是：',self.best_path[self.maxIter-1])
        print('最佳路线历经的城市依次为：', self.best_cities_num)
    def random_seed(self):  # 产生随机的起始点下表，尽量保证所有蚂蚁的起始点不同
        if self.ants_num <= self.cities_num:  # 蚂蚁数 <= 城市数
            self.path_seed[:] = np.random.permutation(range(self.cities_num))[:self.ants_num]
        else:  # 蚂蚁数 > 城市数
            self.path_seed[:self.cities_num] = np.random.permutation(range(self.cities_num))
            temp_index = self.cities_num
            while temp_index + self.cities_num <= self.ants_num:
                self.path_seed[temp_index:temp_index + self.cities_num] = np.random.permutation(range(self.cities_num))
                temp_index += self.cities_num
            temp_left = self.ants_num % self.cities_num
            if temp_left != 0:
                self.path_seed[temp_index:] = np.random.permutation(range(self.cities_num))[:temp_left]

    def display(self):  # 数据可视化展示
        plt.figure(figsize=(6, 10))
        plt.subplot(211)
        plt.plot(self.ants_info, 'g.')
        plt.plot(self.best_path, 'r-', label='history_best')
        plt.xlabel('Iteration')
        plt.ylabel('length')
        plt.legend()
        plt.subplot(212)
        plt.plot(list(city.x for city in self.best_cities), list(city.y for city in self.best_cities), 'g-')
        plt.plot(list(city.x for city in self.best_cities), list(city.y for city in self.best_cities), 'r.')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.savefig('stochastic(10).png', dpi=500)
        plt.show()
        plt.close()


start =time.clock()
ACO()
end = time.clock()
print('程序运行CPU时间为: %s Seconds'%(end-start))