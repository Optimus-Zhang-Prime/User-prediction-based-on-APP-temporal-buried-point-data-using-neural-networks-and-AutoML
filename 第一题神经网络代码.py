import numpy  # 矩阵
import scipy.spatial  # 激活函数1/1+(e^-x)
from matplotlib import pyplot
import scipy.misc
from data import end, label, timex, testend, testtimex
import time
import csv
import sys

class neutralnetwork:  # 神经网络类
    def __init__(self, inputnodes, hiddennodes, outputnodes, learngrates):
        self.inputnode = inputnodes  # 节点数
        self.hiddennode = hiddennodes
        self.outputnode = outputnodes
        # self.weighih = numpy.random.normal(0.0, pow(self.hiddennode, 0.5), (self.hiddennode, self.inputnode))
        self.weighih = numpy.ones([self.hiddennode, self.inputnode], int)
        # 生成权重input—hidden的正态分布矩阵
        self.weighho = numpy.random.normal(0.0, pow(self.outputnode, 0.5), (self.outputnode, self.hiddennode))
        # 生成权重hidden—output的正态分布矩阵
        self.learnrate = learngrates
        self.active_fun = lambda x: numpy.maximum(x , 0)  # 激活函数

    def connect(self, inputs):  # 连接神经网络
        inputinput = numpy.array(inputs, ndmin=2).T  # 矩阵转置
        hiddeninput = numpy.dot(self.weighih, inputinput)  # 隐藏层的输入矩阵  dot为点乘
        hiddenoutput = self.active_fun(hiddeninput)  # 隐藏层的输出矩阵
        outputinput = numpy.dot(self.weighho, hiddenoutput)  # 输出层的输入矩阵
        outputoutput = self.active_fun(outputinput)  # 输出层的输出矩阵
        return outputoutput

    def train(self, inputs, targets):  # 训练神经网络
        input = numpy.array(inputs, ndmin=2).T
        target = numpy.array(targets, ndmin=2).T
        hiddeninput = numpy.dot(self.weighih, input)  # 隐藏层的输入矩阵
        hiddenoutput = self.active_fun(hiddeninput)  # 隐藏层的输出矩阵
        outputinput = numpy.dot(self.weighho, hiddenoutput)  # 输出层的输入矩阵
        outputoutput = self.active_fun(outputinput)  # 输出层的输出矩阵
        outputerror = target - outputoutput  # 输出的误差
        hiddenerror = numpy.dot(self.weighho.T, outputerror)
        # Wj,k的改变量=学习率*k的误差*sigmoid（Ok）*（1-sigmoid（Ok))点乘Oj的转置
        self.weighho += self.learnrate * numpy.dot((outputerror * outputoutput * (1 - outputoutput)),
                                                   numpy.transpose(hiddenoutput))
        self.weighih += self.learnrate * numpy.dot((hiddenerror * hiddenoutput * (1 - hiddenoutput)),
                                                   numpy.transpose(input))
        # 调节权重 其中transpose为转置


time.clock()
# 训练
input_nodes = 123  # 122个输入节点
hidden_nodes = 300
output_nodes = 2  # 两个输出节点
learn_rate = 0.05
n = neutralnetwork(input_nodes, hidden_nodes, output_nodes, learn_rate)
j0 = 0
for index in range(40000000, len(end)):
    yangben = end[index]

    if yangben[0] == 1 & j0 > 1000:  # end中有意义的样本
        onetime = 0
        j0 += 1
        for one in timex:
            if abs(one[0] - index) < 0.1:
                onetime = int(one[1] * 10)
                break
        inputarr = numpy.asfarray(yangben[1:])
        inputarr = numpy.append(inputarr, onetime)
        targets = numpy.zeros(2) + 0.01
        temp = label[index][0]
        targets[temp] = 0.99
        n.train(inputarr, targets)
    elif yangben[0] == 1:
        j0 += 1

print("训练完成", "时间", time.clock())

# 测试成功率
scorescard = []
j = 0
for index1 in range(48876790, len(end)):
    yangben = end[index1]
    if yangben[0] == 1:
        onetime = 0
        for one in timex:
            if abs(one[0] - index1) < 0.1:
                onetime = int(one[1] * 10)
                break
        inputarr = numpy.asfarray(yangben[1:])
        inputarr = numpy.append(inputarr, onetime)
        correctnum = label[index1][0]
        outputarr = n.connect(inputarr)
        outlabel = numpy.argmax(outputarr)
        if outlabel == correctnum:
            scorescard.append(1)
        else:
            scorescard.append(0)
        j += 1
    if j > 1000:
        break
scorescardarr = numpy.asarray(scorescard)
r=scorescardarr.sum() / scorescardarr.size
print("成功率=", r, "时间", time.clock())
if r<0.7:
    print("成功率太低")
    sys.exit()
# 得到测试数据标签
answer = numpy.zeros([15000, 2],int)
i = 0
#
for index1 in range(53452080,55727896):
    yangben = testend[index1]
    if yangben[0] == 1:
        onetime = 0
        for one in testtimex:
            if abs(one[0] - index1) < 0.1:
                onetime = int(one[1] * 10)
                break
        inputarr = numpy.asfarray(yangben[1:])
        inputarr = numpy.append(inputarr, onetime)
        outputarr = n.connect(inputarr)
        outlabel = numpy.argmax(outputarr)
        answer[i][0] = index1
        answer[i][1] = outlabel
        i += 1
        if i%100==0:
            print(i)
with open(r"C:\Users\14531\Desktop\1.csv", "w",newline="")as f:
    writer = csv.writer(f)
    for row in answer:
        writer.writerow(row)

