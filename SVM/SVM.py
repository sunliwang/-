import os
import sys
os.chdir('C:\libsvm_3.22\python')
path = "C:\libsvm_3.22\python"
sys.path.append(path)
from svmutil import *
y, x = svm_read_problem('test.data')
m = svm_train(y[:500], x[:500], '-c 5')
p_label, p_acc, p_val = svm_predict(y[500:], x[500:], m)
print p_acc