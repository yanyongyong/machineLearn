from sklearn.feature_extraction import DictVectorizer
import csv
from sklearn import preprocessing
from sklearn import tree
from sklearn.externals.six import StringIO

# Read in the csv File and put feature in a list of class label
allElectronicsData = open("C:\\Users\\hxjd009\\Desktop\\decisionTree.csv",'rb')
reader = csv.reader(allElectronicsData)
headers = next(reader)
# print headers

featureList = []
labelList = []
# 存放在两个元祖中
for row in reader:
    labelList.append(row[len(row) - 1])
    rowDic = {}
    for i in range(1, len(row) - 1):
        rowDic[headers[i]] = row[i]
    featureList.append(rowDic)

# print featureList
# print labelList

# Vector Feature
vec = DictVectorizer()
dummyX = vec.fit_transform(featureList).toarray()
# print "dummyX:",dummyX
# print vec.get_feature_names()
# print "labelList:"+str(labelList)

lb = preprocessing.LabelBinarizer()
dummyY = lb.fit_transform(labelList)
# print "dummyY:" + str(dummyY)

# using desicionTree for classfication
clf = tree.DecisionTreeClassifier(criterion="entropy")  # 创建一个分类器，entropy决定了用ID3算法
clf = clf.fit(dummyX, dummyY)

# Visulize model
with open("E:\Arduino.dot", "w") as f:
    f = tree.export_graphviz(clf, feature_names=vec.get_feature_names(), out_file=f)

# 预测
oneRowX = dummyX[0, :]
# print "oneRowX:" +str(oneRowX)

newRowX = oneRowX
newRowX[0] = 1
newRowX[2] = 0

predictedY = clf.predict(newRowX)