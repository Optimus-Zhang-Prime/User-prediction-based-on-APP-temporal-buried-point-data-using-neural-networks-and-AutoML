import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
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


end = np.zeros((11000, 123), dtype=np.int8)
k = 0
m = 52263074
index=[5]
for index, f in enumerate(fx):
    temp = idx[index]  # temp为i
    end[k][f] += 1
    end[k][0] = label[temp][0]
    if m != temp:
        k += 1
        m = temp
    if k >= 10000:
        break

X_train, X_test, y_train, y_test = train_test_split(end[:, 1:], end[:, 0],
                                                    train_size=0.95, test_size=0.05)

exported_pipeline = RandomForestClassifier(bootstrap=False, criterion="entropy", max_features=0.8, min_samples_leaf=10,
                                           min_samples_split=10, n_estimators=100)

exported_pipeline.fit(X_train, y_train)
print(exported_pipeline.score(X_test, y_test))




# 读id
idx = np.loadtxt(r"C:\Users\14531\Desktop\test数据处理.csv", delimiter=",", usecols=0)
idx = idx.astype(np.int32)

# 截掉埋点左侧的f
def mystr(s):
    s = s.lstrip("f")
    return s

csv_file = open(r"C:\Users\14531\Desktop\testpoint.CSV")  # 读埋点
csv_reader_lines = csv.reader(csv_file)
date_PyList = []
for one_line in csv_reader_lines:
    date_PyList.append(one_line[0])
date_ndarray0 = list(map(mystr, date_PyList))
fx = list(map(int, date_ndarray0))
csv_file.close()

end = np.zeros((17000, 123), dtype=np.int8)
k = 0
m = 52589302
indexx=[52589302]
for index, f in enumerate(fx):
    temp = idx[index]  # temp为i
    end[k][f] += 1
    if m != temp:
        k += 1
        m = temp
        indexx.append(m)
    if k>14998:
        break
print(len(indexx))
answer=[[],[]]
results = exported_pipeline.predict(end[:11523, 1:])
print(len(results))
answer[0]=list(indexx)
answer[1]=list(results)
answer=np.asfarray(answer)
print("开始存")
answer=np.transpose(np.array(answer))

with open(r"C:\Users\14531\Desktop\2t.csv", "w",newline="")as f:
    writer = csv.writer(f)
    for row in answer:
        writer.writerow(row)
