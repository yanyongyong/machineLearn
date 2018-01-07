import random
import re
import numpy as np

"""
Description:接收一个大字符串并将其解析为字符串列表
    
Params:
    
Returns:
    
Author:
    Yan Yong
Data:
   2018/1/7 14:25 
"""
def textParse(bigString):
    listOfTokens = re.split(r'\W*', bigString)      #将特殊符号作为切分标志进行字符串切分，即非字母、非数字
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]  # 除去单个字母，例如大写的I，其它单词变成小写

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
def createVocabList(sampleData):
    vecData = set([])
    for document in sampleData:
        vecData = vecData | set(document)
    return list(vecData)

"""
Description:根据sample词汇表，将oneWordList向量化，向量的每个元素为1或0
    
Params:
    sample - 词汇表
    oneWordList - 切分的词条列表
Returns:
    returnVec - 文档向量,词集模型
Author:
    Yan Yong
Data:
   2018/1/7 16:39 
"""
def transformVector(sample,oneWordList):
    returnVec = [0] * len(sample)
    for word in oneWordList:
        if word in sample:
            returnVec[sample.index(word)] = 1
    return  returnVec

"""
Description:朴素贝叶斯分类器训练函数
    
Params:
     vector - 训练文档矩阵
     classifyList - 类别标签向量
Returns:
    plVect - 垃圾邮件条件概率数组
    plVect - 非垃圾邮件条件概率数组
    pAbusive - 文档属于垃圾邮件的概率
Author:
    Yan Yong
Data:
   2018/1/7 17:18 
"""
def trainNB0(vector,classifyList):
    vectorLen = len(vector)
    trainLen = len(vector[0])
    p0Num = np.ones(trainLen);p1Num = np.ones(trainLen)  # 创建numpy.ones数组,词条出现数初始化为1，拉普拉斯平滑
    p0Denom = 2.0;p1Denom = 2.0
    pAbusive = sum(classifyList)/float(len(classifyList)) # 文档属于垃圾邮件的概率
    for i in range(vectorLen):
        if classifyList[i] == 1:
            p1Num += vector[i]
            p1Denom += sum(vector[i])
        else:
            p0Num += vector[i]
            p0Denom += sum(vector[i])
    p1Vect = np.log(p1Num/p1Denom)
    p0Vect = np.log(p0Num/p0Denom)
    return p1Vect,p0Vect,pAbusive

"""
Description:朴素贝叶斯分类器邮件分类
    
Params:
    p1Vect - 垃圾邮件条件概率数组
    p0Vect - 非垃圾邮件条件概率数组
    pAbusive - 垃圾邮件先验概率
    waitClssifyWordListVect - 待分类邮件词汇数组向量
Returns:
    
Author:
    Yan Yong
Data:
   2018/1/7 18:02 
"""
def classify(p1Vect, p0Vect, pAbusive,waitClssifyWordListVect):
    p1 = sum(p1Vect * waitClssifyWordListVect) + np.log(pAbusive)
    p0 = sum(p0Vect * waitClssifyWordListVect) + np.log(1 - pAbusive)
    if p1 > p0:
        return 1
    else:
        return 0

def testDeom():
    docList = [];
    classifyList = []
    for i in range(1, 26):
        wordList = textParse(
            open('D:/workSpace/sourceTree/machineLearn/NBM/email/spam/%d.txt' % i, 'r').read())  # 读取每个垃圾邮件，并字符串转换成字符串列表
        docList.append(wordList)
        classifyList.append(1)  # 标记垃圾邮件，1表示垃圾文件
        wordList = textParse(open('D:/workSpace/sourceTree/machineLearn/NBM/email/ham/%d.txt' % i, 'r').read())
        docList.append(wordList)
        classifyList.append(0)
    sample = createVocabList(docList)  # 创建词汇表（不重复）
    trainingSet = list(range(50));
    testSet = []  # 创建存储训练集的索引值的列表和测试集的索引值的列表

    for i in range(10):  # 从50个邮件中，随机挑选出40个作为训练集,10个做测试集
        randIndex = int(random.uniform(0, len(trainingSet)))  # 随机选取索索引值
        testSet.append(trainingSet[randIndex])  # 添加测试集的索引值
        del (trainingSet[randIndex])  # 在训练集列表中删除添加到测试集的索引值
    trainMat = [];
    trainClasses = []  # 创建训练集矩阵和训练集类别标签系向量

    for docIndex in trainingSet:  # 遍历训练集
        trainMat.append(transformVector(sample, docList[docIndex]))  # 将生成的词集模型添加到训练矩阵中
        trainClasses.append(classifyList[docIndex])  # 将类别添加到训练集类别标签系向量中
    p1V, p0V, pSpam = trainNB0(np.array(trainMat), np.array(trainClasses))  # 训练朴素贝叶斯模型
    errorCount = 0  # 错误分类计数

    for docIndex in testSet:  # 遍历测试集
        wordVector = transformVector(sample, docList[docIndex])  # 测试集的词集模型
        if classify(p1V, p0V, pSpam, np.array(wordVector)) != classifyList[docIndex]:  # 如果分类错误
            errorCount += 1  # 错误计数加1
            print("分类错误的测试集：", docList[docIndex])
    print('错误率：%.2f%%' % (float(errorCount) / len(testSet) * 100))

if __name__ == '__main__':
   testDeom()





