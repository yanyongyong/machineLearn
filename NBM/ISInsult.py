import numpy as np


"""
Description:创建实验样本
    
Params:
    
Returns:
    postingList - 实验样本切割的词条
    classVec - 类别标签向量（1为侮辱类，0为非侮辱类）
Author:
    Yan Yong
Data:
   2018/1/6 11:31 
"""
def loadDataSet():
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],  # 切分的词条
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]
    return postingList,classVec

"""
Description:将切分的实验样本词条整理成不重复的词条列表，也就是词汇表
    
Params:
    sampleData -  整理的样本数据集
Returns:
    vecData - 返回不重复的词条列表，也就是词汇表
Author:
    Yan Yong
Data:
   2018/1/6 12:01 
"""
def creaeVocForm(sampleData):
    vecData = set([])
    for document in sampleData:
        vecData = vecData | set(document)
    return list(vecData)

"""
Description:
    
Params:
    vocFrom - 词汇表
    dataSet - 词条
Returns:
    returnVec - 文档向量,词集模型
Author:
    Yan Yong
Data:
   2018/1/6 15:38 
"""
def transformVector(vocFrom,dataSet):
    returnVec = [0] * len(vocFrom)
    for dic in dataSet:
        if dic in vocFrom:
            returnVec[vocFrom.index(dic)] = 1
    return returnVec

"""
Description:朴素贝叶斯分类器训练函数
    
Params:
    vector - 训练文档矩阵
    classVec - 类别标签向量
Returns:
    p0Vect - 侮辱类的条件概率数组
    plVect - 非侮辱类的条件概率数组
    pAbusive - 文档属于侮辱类的概率
Author:
    Yan Yong
Data:
   2018/1/6 19:38 
"""
def probability(vector,classVec):
    vectorLength = len(vector)
    numWords = len(vector[0])
    p0vecSum = 0
    p0matrixSum = np.ones(numWords)
    p1vecSum = 0
    p1matrixSum = np.ones(numWords)
    pAbusive = sum(classVec) / float(vectorLength)  # 文档属于侮辱类的概率
    for i in range(vectorLength):
        if classVec[i] == 1:
            p1matrixSum += vector[i]
            p1vecSum += sum(vector[i])
        else:
            p0matrixSum += vector[i]
            p0vecSum += sum(vector[i])
    plVect = np.log(p1matrixSum/p1vecSum)
    p0Vect = np.log(p0matrixSum/p0vecSum)
    return plVect,p0Vect,pAbusive

"""
Description:朴素贝叶斯分类函数
    
Params:
    plVect - 侮辱类的条件概率数组
    plVect - 非侮辱类的条件概率数组
    waitClssifyWord - 待分类词组
    dataSet - 词条
    insultPro - 侮辱先验概率
Returns:
    
Author:
    Yan Yong
Data:
   2018/1/6 20:20 
"""
def classify(plVect,p0Vect,dataSet,waitClssifyWord,insultPro):
    pl = 1
    p0 = 1
    for word in waitClssifyWord:
        if word in dataSet:
            pl *= plVect[dataSet.index(word)]
            p0 *= p0Vect[dataSet.index(word)]
        pl = pl + np.log(insultPro)
        p0 = p0 + np.log(1-insultPro)
    if pl > p0:
        return 0
    else:
        return 1

if __name__ == '__main__':
    waitClssifyWord = ['my', 'food','dog']
    trainMat = []
    postingList,classVec = loadDataSet()
    dataSet = creaeVocForm(postingList)
    for ds in postingList:
        trainMat.append(transformVector(dataSet,ds))
    plVect,p0Vect,pAbusive = probability(trainMat,classVec)
    print(dataSet)
    print(plVect)
    print(p0Vect)
    print(pAbusive)
    classify = classify(plVect,p0Vect,dataSet,waitClssifyWord,pAbusive)
    if classify == 0:
        print('非侮辱词')
    else:
        print('侮辱词')