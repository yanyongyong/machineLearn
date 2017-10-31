import numpy as np #算矩阵
import pylab as pl #图形工具
from sklearn import svm

#we create 40 separable points
np.random.seed(0)
#20代表通过正态分布随机产生2维的20个点，(20,2)20行2列
X = np.r_[np.random.randn(20,2) - [2,2],np.random.randn(20,2) + [2,2]]
#归类标记,前20个点标记为0
Y = [0] * 20 + [1] * 20

# fit the model
clf = svm.SVC(kernel='linear')
clf.fit(X,Y) #以矩阵的方式传入，进行回归计算

#二维的超平面公式：W0X + W1Y + W3 = 0 → Y = -（W0/W1）X - W3/W1
#超平面模型上面已经建好了，现在要做的是画出来
#get the separating(分离) hyperplane(超平面)
w = clf.coef_[0] #存放回归系数，W已知是二维的
print(w)
#斜率a
a = -w[0]/w[1]
xx = np.linspace(-5,5) #-5到5之间产生连续的xx的值
yy = a * xx - (clf.intercept_[0]/w[1]) #intercept_存放截距,就是W3

#plot the parallels to separation hyperplane that pass through the suppor vectors
#取到第一个支持向量（下面）
b = clf.support_vectors_[0]
yy_down = a * xx + (b[1] - a *b[0])
#取第二个（上面）的支持向量
b =clf.support_vectors_[-1]#-1是python中特殊用法，比如去list中最后一个数据就用-1
yy_up = a * xx + (b[1] - a * b[0])

#画线
pl.plot(xx,yy,'k-')
pl.plot(xx,yy_down,'k--')
pl.plot(xx,yy_up,'k--')

#标记支持向量
pl.scatter(clf.support_vectors_[:,0],clf.support_vectors_[:,1],s=80,facecolors='red')
#获取支持向量
print(clf.support_vectors_)
print(clf.support_vectors_[:,0])
print(clf.support_vectors_[:,1])
#画点
pl.scatter(X[:,0],X[:,1],c=Y,cmap=pl.cm.Paired)
pl.axis('tight')
pl.show()