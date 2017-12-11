from sklearn.feature_extraction import DictVectorizer
import csv
from sklearn import preprocessing
from sklearn import tree
from sklearn.externals.six import StringIO

#决策树实例
decisionTreeData = open('C:\\Users\\hxjd009\\Desktop\\decisionTree.csv',encoding='utf-8')
reader = csv.reader(decisionTreeData)
headers = next(reader)
print(headers)
# for row in reader:
#     print(row)

featureList = []
lableList = []
for rows in reader:
    lableList.append(rows[len(rows) - 1])#获取标签
    print(lableList)
    rowDict = {}
    for i in range(1,len(rows) - 1):
        rowDict[headers[i]] = rows[i]#获取特征
    featureList.append(rowDict)
# print(featureList)

#特征的装换
vec = DictVectorizer()
dummyX = vec.fit_transform(featureList).toarray()
# print("dummyX:",dummyX)
# print(vec.get_feature_names())
# print(str(lableList))

#结果的转换(转化为0,1)
lb = preprocessing.LabelBinarizer()
dummyY = lb.fit_transform(lableList)
# print("dummyY:",dummyY)

clf = tree.DecisionTreeClassifier(criterion='entropy')
clf = clf.fit(dummyX,dummyY)
# print("clf:"+str(clf))

#dot文件转化为pdf文件
with open('E:\Arduino.dot','w') as f:
    f = tree.export_graphviz(clf,feature_names = vec.get_feature_names(),out_file = f)

#输入数据预测
oneRowX = dummyX[0]
print("newRowX:",oneRowX)

newRowX = oneRowX
newRowX[0] = 0
newRowX[1] = 0

# print("newRowX:"+str(newRowX))
# print("shape:"+str(newRowX.shape))
# print("reshape:"+str(newRowX.reshape(1, -1)))
predictedY = clf.predict(newRowX.reshape(1, -1))
print("predictedY:"+str(predictedY))
