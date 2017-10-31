import numpy as np #算矩阵
import pylab as pl #图形工具
from sklearn import svm

#we create 40 separable points
np.random.seed(0)
#20代表通过正态分布随机产生2维的20个点
x = np.r_[np.random.randn(20,2) - [2,2],np.random.randn(20,2) + [2,2]]
#归类标记,前20个点标记为0
y = [0] * 20 + [1] * 20

# fit the model
clf = svm.SVC(kernel='linear')
clf.fit(x,y)

