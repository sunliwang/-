import math
import random
import numpy as np
random.seed(0)
#生成ab之间的随机数
def rand(a, b):
    return (b - a) * random.random() + a
#创建矩阵 m*n
def make_matrix(m, n, fill=0.0):
    mat = []
    for i in range(m):
        mat.append([fill] * n)
    return mat
#sigmod函数
def sigmod(x):
    return 1.0 / (1.0 + math.exp(-x))
#sigmod函数求导
def sigmod_derivate(x):
    return x * (1 - x)
# 导入文件
def loaddata(filename):
    fr = open(filename)
    line = fr.readlines()
    numberofline = len(line)
    data = np.zeros((numberofline, 8))
    labels = np.zeros((numberofline, 1))
    index = 0
    for context in line:
        context = context.strip().split(',')
        data[index,:] = context[0:8]
        labels[index, 0] = context[-1]
        index += 1
    return data, labels
#列的归一化
def unification(datamatrix):
    minvalues = datamatrix.min(0)
    maxvalues = datamatrix.max(0)
    ranges = maxvalues - minvalues
    numberofline = datamatrix.shape[0]
    normmatrix = datamatrix - np.tile(minvalues, (numberofline, 1))
    normmatrix = normmatrix / np.tile(ranges, (numberofline, 1))
    return normmatrix

#创建一个神经网络的类
class BPNeuralNetwork:
    def __init__(self):
        self.stat=0
        self.input_n = 0 
        self.hidden_n = 0
        self.hidden_n1=0
        self.output_n = 0
        self.input_cells = [] #输入层
        self.hidden_cells = [] #隐藏层
        self.hidden_cells1=[]
        self.output_cells = [] #输出层
        self.input_weights = [] #输入层的权重
        self.hidden_weights=[]
        self.output_weights = [] #输出层的权重
        self.input_correction = [] #输入层的偏置值
        self.hidden_correction=[]
        self.output_correction = [] #输出层的偏置值

    def setup(self, ni, nh,nh1, no):
        self.input_n = ni + 1 #输入参数的个数
        self.hidden_n = nh #隐层参数个数
        self.hidden_n1=nh1
        self.output_n = no #输出参数的个数
        #初始化神经网络的三层
        self.input_cells = [1.0] * self.input_n
        self.hidden_cells = [1.0] * self.hidden_n
        self.hidden_cells1=[1.0] * self.hidden_n
        self.output_cells = [1.0] * self.output_n
        # 初始化权重
        self.input_weights = make_matrix(self.input_n, self.hidden_n)
        self.hidden_weights=make_matrix(self.hidden_n,self.hidden_n1)
        self.output_weights = make_matrix(self.hidden_n1, self.output_n)
        for i in range(self.input_n):
            for h in range(self.hidden_n):
                self.input_weights[i][h] = rand(-0.2, 0.2)#初始化输入与隐层之间的权重
        for i in range(self.hidden_n):
            for h in range(self.hidden_n1):
                self.hidden_weights[i][h] = rand(-0.2, 0.2)
        for h in range(self.hidden_n1):
            for o in range(self.output_n):
                self.output_weights[h][o] = rand(-2.0, 2.0)#初始化隐层与输出之间的权重
        self.input_correction = make_matrix(self.input_n, self.hidden_n)
        self.hidden_correction=make_matrix(self.hidden_n,self.hidden_n1)
        self.output_correction = make_matrix(self.hidden_n1, self.output_n)
#预测
    def predict(self, inputs,label,operation=False):
        for i in range(self.input_n - 1):
            self.input_cells[i] = inputs[i]
        for j in range(self.hidden_n):
            total = 0.0
            for i in range(self.input_n):
                total += self.input_cells[i] * self.input_weights[i][j]+input_correction[i][j]
            self.hidden_cells[j] = sigmod(total)
        for j in range(self.hidden_n1):
            total=0.0
            for i in range(self.hidden_n): 
                total+=hidden_cells[i][j]*self.hidden_weights[i][j]+hidden_correction[i][j]
            self.hidden_cells1[j]=sigmod(total)
        for k in range(self.output_n):
            total = 0.0
            for j in range(self.hidden_n1):
                total += self.hidden_cells1[j] * self.output_weights[j][k]+output_correction[i][j]
            self.output_cells[k] = sigmod(total)
        if (self.output_cells[0]>0.5 and label[0]==1) or (self.output_cells[0]<=0.5 and label[0]==0):
            if operation==True:
                self.stat=self.stat+1
            return 'correct prediction'
        else:
            return 'error prediction'
#反向传播
    def back_propagate(self, case, label, learn, correct):
        self.predict(case,label)
        output_deltas = [0.0] * self.output_n
        for o in range(self.output_n):
            error = label[o] - self.output_cells[o]
            output_deltas[o] = sigmod_derivate(self.output_cells[o]) * error


        hidden_deltas1 = [0.0] * self.hidden_n1
        for h in range(self.hidden_n1):
            error = 0.0
            for o in range(self.output_n):
                error += output_deltas[o] * self.output_weights[h][o]
            hidden_deltas1[h] = sigmod_derivate(self.hidden_cells1[h]) * error

#deltas是为了描述误差，修改权值和偏置值
        hidden_deltas = [0.0] * self.hidden_n
        for h in range(self.hidden_n):
            error = 0.0
            for o in range(self.hidden_n1):
                error += hidden_deltas1[o] * self.hidden_weights[h][o]
            hidden_deltas[h] = sigmod_derivate(self.hidden_cells1[h]) * error
#--------------------------------------------------------------------------------------------------
        for h in range(self.hidden_n1):
            for o in range(self.output_n):
                change = output_deltas[o] * self.hidden_cells1[h]
                self.output_weights[h][o] += learn * change
                self.output_correction[h][o] +=learn*change
#用后一层的误差来更新该层与后一层的权值
        for i in range(self.hidden_n):
            for h in range(self.hidden_n1):
                change = hidden_deltas1[h] * self.hidden_cells[i]
                self.hidden_weights[i][h] += learn * change
                self.hidden_correction[i][h] +=learn*change

        for i in range(self.input_n):
            for h in range(self.hidden_n):
                change = hidden_deltas[h] * self.input_cells[i]
                self.input_weights[i][h] += learn * change
                self.input_correction[i][h] +=learn*change




        error = 0.0
        for o in range(len(label)):
            error += 0.5 * (label[o] - self.output_cells[o]) ** 2
        return error
#训练
    def train(self, cases, labels, learn=0.05, correct=0.1):
#        for i in range(10):#这里我就利用重复的数据来进行训练
            error = 0.0
            for i in range(len(cases)):
                label = labels[i]
                case = cases[i]
                error += self.back_propagate(case, label, learn, correct)
#测试数据
    def Deal(self):
        loaddata('breast_cancer.data')
        data1, labels = loaddata('breast_cancer.data')
        for i in range(len(data1[0])):
            temp=unification(data1[:,i])
            for j in range(len(data1)):
                data1[j,i]=temp[j,0]
        
        self.setup(8,8, 8, 1)
        self.train(data1[0:500], labels[0:500],0.4, 0.2)
        for i in range(500,768):
            print(self.predict(data1[i],labels[i],True))
        print('Accuracy:%f'%(float(self.stat)/268))
        


if __name__ == '__main__':
    nn = BPNeuralNetwork()
    nn.Deal()