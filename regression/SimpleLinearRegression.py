import numpy as np
#简单的线性回归
def fitSLR(x,y):
    n = len(x)
    denominator = 0 #分母
    numerator = 0 #分子
    for i in range(0,n):
        numerator += (x[i]- np.mean(x))*(y[i] - np.mean(y))
        denominator += (x[i] - np.mean(x))**2
    b1 = numerator/float(denominator)
    b0 = np.mean(y) - b1*np.mean(x)
    # b0 = np.mean(y)/float(np.mean(x))
    return b0, b1

def predict(x,bo,b1):
    return bo + x*b1

x = [1,3,2,1,3]
y = [14,24,18,17,27]

b0,b1 = fitSLR(x,y)

x_test = 8
y_test = predict(8,b0,b1)

print(y_test)