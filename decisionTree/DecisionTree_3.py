from math import log

"""
Description:
    创建数据集
Params:
    无
Returns:
    dataSet - 数据集
    labels - 分类属性
Author:
    Yan Yong
Data:
   2017/12/23 18:19 
"""
def creatDataSet():
    dataSet = [[0, 0, 0, 0, 'no'],  # 数据集
               [0, 0, 0, 1, 'no'],
               [0, 1, 0, 1, 'yes'],
               [0, 1, 1, 0, 'yes'],
               [0, 0, 0, 0, 'no'],
               [1, 0, 0, 0, 'no'],
               [1, 0, 0, 1, 'no'],
               [1, 1, 1, 1, 'yes'],
               [1, 0, 1, 2, 'yes'],
               [1, 0, 1, 2, 'yes'],
               [2, 0, 1, 2, 'yes'],
               [2, 0, 1, 1, 'yes'],
               [2, 1, 0, 1, 'yes'],
               [2, 1, 0, 2, 'yes'],
               [2, 0, 0, 0, 'no']]
    labels = ['年龄', '有工作', '有自己的房子', '信贷情况']  # 分类属性
    return dataSet, labels  #返回数据集和分类属性

"""
Description:
    计算香农熵（信息熵）
Params:
    dataSet - 数据集
Returns:
    
Author:
    Yan Yong
Data:
   2017/12/23 18:11 
"""
def calculateEntropy(dataSet):
    lableCounts = {}                             #保存每个标签(Label)出现次数的字典
    dataLen = len(dataSet)
    for i in dataSet:
        currentLable = i[-1]                     #提取标签(Label)信息(取最后一个数据)
        if currentLable not in lableCounts.keys():
            lableCounts[currentLable] = 0
        lableCounts[currentLable] += 1
    shannonEnt = 0.0
    for key in lableCounts:                     #经验熵(香农熵)
        prob = lableCounts[key]/dataLen
        shannonEnt -= prob*(log(prob,2))
    return shannonEnt

"""
Description:
    分割数据集
Params:
    dataSet - 数据集
    axis - 划分数据集的特征
    value - 需要返回的特质值
Returns:
    retDataSet - 挑选后的数据集
Author:
    Yan Yong
Data:
   2017/12/24 13:49 
"""
def splitDataSet(dataSet,axis,value):
    retDataSet = []
    for i in dataSet:
        if i[axis] == value:
            retDataSet.append(i)
    return retDataSet

"""
Description: 
    计算信息增益
Params:
    dataSet - 数据集
Returns:
    
Author:
    Yan Yong
Data:
   2017/12/23 21:04 
"""
def informationGain(dataSet):
    labelsGain = {}
    dataSetLeg = len(dataSet[0]) - 1
    dataSetLen = len(dataSet)
    shannonEntYesNo = calculateEntropy(dataSet)
    for j in range(dataSetLeg):
        featList = [example[j] for example in dataSet]  # 获取dataSet的第i个所有特征
        uniqueVals = set(featList)
        shannonEnt = 0.0
        for value in uniqueVals:                        #计算信息增益
            retDataSet = splitDataSet(dataSet,j,value)
            shannonEnt += (len(retDataSet)/dataSetLen) * calculateEntropy(retDataSet)
        labelsGain[j] = shannonEntYesNo - shannonEnt
    return labelsGain




if __name__ == '__main__':
    dataSet,labels = creatDataSet()
    # print(calculateEntropy(dataSet))
    labelsGain = informationGain(dataSet)
    print(labelsGain)