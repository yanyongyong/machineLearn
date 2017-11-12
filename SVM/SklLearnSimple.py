from sklearn import svm

#SVM画简单的超平面（3个点）

#SVM的超平面
x = [[2,0],[1,1],[2,3]]
y = [0,0,1] #类别 [2.0]对应的类别为0，[1,1]对应的类别为0
clf = svm.SVC(kernel='linear')
clf.fit(x,y)

print(clf)

# get support vectors(获取支持向量)
print(clf.support_vectors_)

# get indices of support vectors（获取支持向量在数组中的位置）
print(clf.support_)

# get number of support vectors each class(分类分别找到了几个支持向量)
print(clf.n_support_)

#预测点的分类
print(clf.predict([[1,1]]))
