import csv
import numpy as np
from PIL import Image
from scipy.misc import imsave
import matplotlib.pyplot as plt


#读CSV格式的文件
# my_matrix = np.loadtxt(open("E:/MachineLearning/KNN/data/train.csv","rb"),delimiter=",",skiprows=0)
#写
# np.savetxt('new.csv', my_matrix, delimiter = ',')

def readCSV():
    imageData = open("E:/MachineLearning/KNN/data/train.csv",encoding="utf-8")
    reader = csv.reader(imageData)
    header = next(reader)
    print(header)

###################################一种方式（PIL）##########################################
#将图片转化为矩阵
def ImageToMatrix(filename):
    image = Image.open(filename)
    # image.show()
    width,height = image.size
    # 将图像转换为“L”图像
    image = image.convert('L')
    data = image.getdata()
    data = np.matrix(data,dtype='float')/256.0
    new_data = np.reshape(data,(width,height))
    return new_data
#将矩阵转化为图片
def MatrixToImage(data):
    data = data*256
    new_im = Image.fromarray(data.astype(np.uint8))
    return new_im
# data = ImageToMatrix('E:/yanyong/meitu/2016.1.jpg')
# print(data)
# new_im = MatrixToImage(data)
# plt.imshow(data,cmap=plt.cm.gray,interpolation='nearest')
# new_im.show()
# new_im.save('lena_1.bmp')


##############################################第二种方式######################################################
#提取csv数据的第二行数据
def readerCSV(file):
    with open(file, 'rt') as csvfile:
        reader = csv.reader(csvfile)
        for i, rows in enumerate(reader):
            if i == 1:
                row = rows
        return row


#提取除第一个以外的数据
def extract(row):
    save = []
    for i in range(1,len(row)):
        # save.append(int(row[i]))
        save.append(row[i])
    return save

#矩阵转化为图片
def matrixToImage(outFile,matrix):
    # x = np.random.random((600,800,3))
    imsave(outFile,matrix)

#n维转化为n维矩阵
def arrayToMatrix(arrayData,row,line):
    matrixData = np.array(arrayData).reshape(row, line)
    return matrixData

#欧氏距离
def euclideanMetric(vector1,vector2):
    # return np.linalg.norm(vector1 - vector2)
    return np.sqrt(np.sum(np.square(vector1 - vector2)))



def runMatricToImage():
    newImgData = readerCSV('E:/MachineLearning/KNN/data/train.csv')
    save = extract(newImgData)
    print(len(save))
    matrixData = arrayToMatrix(save, 28, 28)
    matrixToImage('1.jpg', matrixData)

#####################################图片识别########################################
#训练数据
def creatDataSet(file):
    imageData = []
    labels = []
    with open(file, 'rt') as csvfile:
        reader = csv.reader(csvfile)
        for i, rows in enumerate(reader):
            labels.append(rows[0])
            imageData.append(rows)
        return imageData,labels

#预测数据
def preDataSet(file):
    preImageData = []
    with open(file, 'rt') as csvfile:
        reader = csv.reader(csvfile)
        for i, rows in enumerate(reader):
            preImageData.append(rows)
        del preImageData[0]
        preImageData = manyDimToInt(preImageData)
        return preImageData

#Sring类型数组转int类型
def stringToInt(array):
    return [int(i) for i in array]

#多维String类型数组转int型类型
def  manyDimToInt(dimArray):
    newData = []
    for i in dimArray:
        newData.append(stringToInt(i))
    return newData

#训练数据和预测数据比较
def preNumber(preData,trainData):
    preResult = []
    # for i in range(28000):
    for j in range(100):
        del trainData[j][0]
        result = int(euclideanMetric(np.array(preData),np.array(trainData[j])))
        preResult.append(result)
    return preResult


# #删除数组的第一行
# def delOneRow(array):
#     del array

#删除有特定字符的
# def delSpecificStr(string):


if __name__ == '__main__':

    #################样本数据#############################
    imageData, labels = creatDataSet("E:/MachineLearning/KNN/data/train.csv")
    #删除有label
    labels.remove('label')
    # print(labels)
    #删除imageData[0]行
    del imageData[0]
    newArray = manyDimToInt(imageData)

    #######################预测数据######################
    preImageData = preDataSet("E:/MachineLearning/KNN/data/test.csv")
    # print(len(preImageData[0]))
    # print(len(newArray[1]))
    # print(newArray[1])

    # a = np.array(newArray[0])
    # b = np.array(newArray[2])
    # distance = euclideanMetric(a, b)
    distance = preNumber(preImageData[0],newArray)

    distance.sort()
    print(distance)













