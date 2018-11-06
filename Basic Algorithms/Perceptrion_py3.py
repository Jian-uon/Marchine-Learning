# -*- coding: UTF-8 -*-

# **********************************************************
# * Author        : HuJian
# * Email         : swjtu_hj@163.com
# * Create time   : 2018-11-06 17:05
# * Last modified : 2018-11-06 17:05
# * Filename      : Perceptrion2.py
# * Description   : python3
# **********************************************************
from functools import reduce
import numpy as np

class Perceptron():
    def __init__(self, input_vec, label, activator):
        '''初始化参数'''
        self.input_vec = input_vec
        self.label = label
        self.activator = activator
        '''activator是激活函数，是 一个  函数'''
        self.weight = np.random.rand(2)
        '''初始化每个权重'''
        self.bias = np.random.rand()
        '''初始化偏执项'''

    def __str__(self):
        '''打印学习到的权重w和偏执b'''
        return 'weights\t:%s\nbias\t:%s\n' % (self.weight, self.bias)

    def predict(self,temp_vec):
        '''输入参数进行预测'''
        temp_a =list(map(lambda x: x[0] * x[1], zip(temp_vec, self.weight)))
        net = sum(temp_a)
        return self.activator(net+self.bias)


    def train(self,iter_count,rate):
        count = 0
        while True:
            e_sum=0
            for i in range(len(self.input_vec)):
               predict_v=self.predict(temp_vec=self.input_vec[i])
               delta = self.label[i]-predict_v
               e_sum+=delta
               if delta!=0:
                   self.weight = list(map(lambda x:x[1]+rate*delta*x[0],zip(self.input_vec[i],self.weight)))
                   self.bias +=rate*delta
            count+=1
            if count>iter_count:
                print('迭代：%i 次 delta: %f' % (count, e_sum))
                print(self.__str__())
                break

    def forecast(self,forecast_vec):
        '''预测部分'''
        forecast_result= [0.0 for i in range(len(forecast_vec))]
        for i in range(len(forecast_vec)):
            forecast_result[i] = self.predict(forecast_vec[i])
            print('预测向量：%s %s\t预测结果：%s' % (forecast_vec[i][0],forecast_vec[i][1],forecast_result[i]))


def f(x):
    '''激活函数'''
    return 1 if x > 0 else 0


if __name__ == '__main__':
    train_vector = [[1, 1], [0, 0], [1, 0], [0, 1]]
    train_label = [1, 0, 0, 0]

    test_vector = [[1, 1], [0, 0], [1, 0], [0, 1]]

    iter_count = 100
    rate = 0.01

    p = Perceptron(train_vector, train_label, f)
    p.train(iter_count, rate)
    p.forecast(test_vector)
