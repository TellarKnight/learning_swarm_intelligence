import numpy as np
import matplotlib.pyplot as plt
import time

# 建立“蚂蚁”类
class Ant(object):
    def __init__(self, path):
        self.path = path  # 蚂蚁当前迭代整体路径
        self.length = self.calc_length(path)  # 蚂蚁当前迭代整体路径长度

    def calc_length(self, path_):
        length_ = 0
        for i in range(len(path_) - 1):
            delta = (path_[i].x - path_[i + 1].x, path_[i].y - path_[i + 1].y)
            length_ += np.linalg.norm(delta)
        return length_


    def calc_len(A, B):  # 静态方法计算城市A与城市B之间的距离
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


# 构建带精英策略的蚁群算法的主体
class ACO(object):
    def __init__(self, ant_num=50, maxIter=300, alpha=1, beta=5, rho=0.5, Q=100, e=5):
        self.ants_num = ant_num  # 蚂蚁个数
        self.maxIter = maxIter  # 蚁群最大迭代次数
        self.alpha = alpha  # 信息启发式因子
        self.beta = beta  # 期望启发式因子
        self.rho = rho  # 信息素挥发速度
        self.Q = Q  # 信息素强度
        self.e = e  #  精英蚂蚁数
        ###########################
        #self.deal_data('coordinates-of-cities.dat')  # 提取所有城市的坐标信息
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
        self.cities_num = len(temp_list)  # 获取城市个数
        self.cities = list(City(float(item[0]), float(item[1])) for item in temp_list)  # 构建城市列表
        self.city_dist_mat = np.zeros((self.cities_num, self.cities_num))  # 构建城市距离矩阵
        self.best_cities_num = np.zeros(self.cities_num+1)  # 初始化最佳路径城市列表
        for i in range(self.cities_num):
            A = self.cities[i]
            for j in range(i, self.cities_num):
                B = self.cities[j]
                self.city_dist_mat[i][j] = self.city_dist_mat[j][i] = Ant.calc_len(A, B)
        self.phero_mat = (10**(-6))*np.ones((self.cities_num, self.cities_num))  # 初始化信息素矩阵
        self.eta_mat = 1 / (self.city_dist_mat + np.diag([np.inf] * self.cities_num))  # 初始化启发函数矩阵

    def solve(self):
        iterNum = 0  # 当前迭代次数
        while iterNum < self.maxIter:
            self.random_seed()  # 使整个蚁群产生随机的起始点
            delta_phero_mat = np.zeros((self.cities_num, self.cities_num))  # 初始化每次迭代后信息素矩阵的增量
            delta_e_phero_mat = np.zeros((self.cities_num, self.cities_num))  # 初始化每次迭代后由精英蚂蚁引起信息素矩阵的增量
            ##########################################################################
            for i in range(self.ants_num):
                city_index1 = self.path_seed[i]  # 每只蚂蚁访问的第一个城市下标
                ant_path = Path(self.cities[city_index1])  # 记录每只蚂蚁访问过的城市
                tabu = [city_index1]  # 记录每只蚂蚁访问过的城市下标，禁忌城市下标列表
                non_tabu = list(set(range(self.cities_num)) - set(tabu))
                for j in range(self.cities_num - 1):  # 对余下的城市进行访问
                    up_proba = np.zeros(self.cities_num - len(tabu))  # 初始化状态迁移概率的分子
                    for k in range(self.cities_num - len(tabu)):
                        up_proba[k] = np.power(self.phero_mat[city_index1][non_tabu[k]], self.alpha) * (np.power(self.eta_mat[city_index1][non_tabu[k]], self.beta))
                    proba = up_proba / sum(up_proba)  # 每条可能子路径上的状态迁移概率
                    while True:  # 提取出下一个城市的下标
                        random_num = np.random.rand()
                        index_need = np.where(proba > random_num)[0]
                        if len(index_need) > 0:
                            city_index2 = non_tabu[index_need[0]]
                            break
                    ant_path.add_path(self.cities[city_index2])
                    tabu.append(city_index2)
                    non_tabu = list(set(range(self.cities_num)) - set(tabu))
                    city_index1 = city_index2
                self.ants_info[iterNum][i] = Ant(ant_path.path).length
                if iterNum == 0 and i == 0:  # 完成对最佳路径城市的记录
                    self.best_cities = ant_path.path
                else:
                    if self.ants_info[iterNum][i] < (Ant(self.best_cities).length-0.05): self.best_cities = ant_path.path
                tabu.append(tabu[0])  # 每次迭代完成后，使禁忌城市下标列表形成完整闭环
                for l in range(self.cities_num):
                    delta_phero_mat[tabu[l]][tabu[l + 1]] += self.Q / self.ants_info[iterNum][i]
                for l in range(self.cities_num):
                    delta_phero_mat[tabu[l+1]][tabu[l]] = delta_phero_mat[tabu[l]][tabu[l + 1]]
            self.best_path[iterNum] = Ant(self.best_cities).length
            for u in range(0, self.cities_num):
                for z in range(0, self.cities_num):
                    if self.cities[z] == self.best_cities[u]:
                        self.best_cities_num[u] = z
                        break
            self.best_cities_num[self.cities_num] = self.best_cities_num[0]
            '''
            i2=0
            for w in range(0, self.cities_num):
                while i2 < self.cities_num :
                    for j2 in range(0, self.cities_num):
                        if (i2 == self.best_cities_num[w] and j2 == self.best_cities_num[w + 1]):
                            #delta_e_phero_mat[i2][j2] = delta_phero_mat[j2][i2] = self.Q / self.best_path[iterNum]
                            delta_e_phero_mat[i2][j2] = self.Q / self.best_path[iterNum]
                            i2=i2+1
                            break
                    break
             '''

            for w in range(self.cities_num):
               delta_e_phero_mat[int(self.best_cities_num[w])][int(self.best_cities_num[w+1])] = self.Q / self.best_path[iterNum]
            for w in range(self.cities_num):
               delta_e_phero_mat[int(self.best_cities_num[w + 1])][int(self.best_cities_num[w])] = delta_e_phero_mat[int(self.best_cities_num[w])][int(self.best_cities_num[w+1])]

            self.update_phero_mat(delta_phero_mat,delta_e_phero_mat)  # 更新信息素矩阵
            iterNum += 1

        for i in range(self.maxIter):
            if(self.best_path[i] == self.best_path[self.maxIter-1]):
                print('算法最少使用%d次迭代得到最优解'%(i+1))
                break

        print('最佳路线历经的城市依次为：',self.best_cities_num)
        print('达到最大迭代次数后得到的最短路径长度是：',self.best_path[self.maxIter-1])

        #for i2 in range(self.cities_num):
           #print(self.ants_info[self.maxIter-1][i2-1])
    def update_phero_mat(self, delta,delta_e):
        self.phero_mat = (1 - self.rho) * self.phero_mat + delta+self.e*delta_e


    def random_seed(self):  # 产生随机的起始点下标，尽量保证所有蚂蚁的起始点不同
        self.path_seed[:] = np.random.permutation(range(self.cities_num))[:self.ants_num]


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
        plt.savefig('AS-TSP-50-1-5-0.5-100-5(10).png', dpi=500)
        plt.show()
        plt.close()

start =time.clock()
ACO()
end = time.clock()
print('程序运行CPU时间为: %s Seconds'%(end-start))