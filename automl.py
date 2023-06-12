import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from tpot import TPOTClassifier
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
print(end[:][1])

X_train, X_test, y_train, y_test = model_selection.train_test_split(end[:, 1:], end[:, 0],
                                                                    train_size=0.75, test_size=0.25)
pipeline_optimizer = TPOTClassifier(generations=5, population_size=20, cv=5,
                                    random_state=42, verbosity=2)
pipeline_optimizer.fit(X_train, y_train)
print(pipeline_optimizer.score(X_test, y_test))
pipeline_optimizer.export("code.py")
