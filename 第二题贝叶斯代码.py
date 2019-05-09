import numpy as np
import pandas as pd

from sklearn import datasets, naive_bayes
from sklearn.model_selection import train_test_split
import csv

# 读id
idx = np.loadtxt(r"C:\Users\14531\Desktop\数据处理.CSV", delimiter=",", usecols=0)
idx = idx.astype(np.int32)


# 截掉埋点左侧的f
def mystr(s):
    s = s.lstrip("f")
    return s


csv_file = open(r"C:\Users\14531\Desktop\point.CSV")  # 读埋点
csv_reader_lines = csv.reader(csv_file)
date_PyList = []
for one_line in csv_reader_lines:
    date_PyList.append(one_line[0])
date_ndarray0 = list(map(mystr, date_PyList))
fx = list(map(int, date_ndarray0))
csv_file.close()

label = np.zeros((60000000, 1), dtype=np.int8)  # 存label的二维数组
labelx = np.loadtxt(r"C:\Users\14531\Desktop\Label.csv", delimiter=",", usecols=(0, 1))
for i in range(len(labelx)):
    temp = int(labelx[i][0])
    label[temp][0] = labelx[i][1]

# 创建二维数组，id为行数，埋点为列数
end = np.zeros((10000, 123), dtype=np.int8)
k = 0
m = 52263074
for index, f in enumerate(fx):
    temp = idx[index]  # temp为id
    end[k][f] += 1
    end[k][0] = label[temp][0]
    if m != temp:
        k += 1
        m = temp
    if k >= 10000:
        break

x_train, x_test, y_train, y_test = train_test_split(end[:, 1:], end[:, 0],
                                                    train_size=0.75, test_size=0.25)

cls = naive_bayes.GaussianNB()
cls.fit(x_train, y_train)
print('高斯贝叶斯分类器')
print('Training Score: %.2f' % cls.score(x_train, y_train))
print('Testing Score: %.2f' % cls.score(x_test, y_test))
