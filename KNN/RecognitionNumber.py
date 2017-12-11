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

save = []
#提取除第一个以外的数据
def extract(row):
    for i in range(1,len(row)):
        save.append(int(row[i]))

#矩阵转化为图片
def matrixToImage(outFile,matrix):
    # x = np.random.random((600,800,3))
    imsave(outFile,matrix)

#n维转化为n维矩阵
def arrayToMatrix(arrayData,row,line):
    matrixData = np.array(arrayData).reshape(row, line)
    return matrixData

newImgData = readerCSV('E:/MachineLearning/KNN/data/train.csv')
extract(newImgData)
print(len(save))
matrixData = arrayToMatrix(save,28,28)
matrixToImage('1.jpg',matrixData)








