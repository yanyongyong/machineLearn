from math import log

#算香农熵（信息熵）

def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        #去最后一个
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            print("========" + currentLabel)
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
        #INT型输出
        print("-----------",labelCounts)
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob * log(prob,2)
    return shannonEnt

def createDataSet():
    dataSet = [[1,1,'yes'],[1,1,'yes'],[1,1,'yes'],[1,0,'no'],[0,1,'no'],[0,1,'no']]
    lables = ['no surfacing','flippers']
    return dataSet,lables

if __name__ == '__main__':
    dataSet,labels = createDataSet();
    print(dataSet)
    print(labels)
    result = calcShannonEnt(dataSet)
    print(result)