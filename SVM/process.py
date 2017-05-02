import os
import numpy as np
import pandas as pd
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
datas1,labels=loaddata('breast_cancer.data')
datas=unification(datas1)
datas=datas.astype(np.str)
for j in range(len(datas[0])):
    for i in range(len(datas)):
        datas[i,j]=str(j+1)+':'+str(datas[i,j])
data=np.column_stack((labels,datas))
fw=open('test.data','w')
for i in range(len(data)):
    for j in range(len(data[0])):
        fw.write(data[i,j])
        fw.write('\t')
    fw.write('\n')
fw.close()