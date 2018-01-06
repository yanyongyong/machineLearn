import numpy as np

def test():
    dataSet = [[0, 0, 0, 0],
               [0, 0, 0, 1],
               [0, 1, 0, 1],
               [0, 1, 1, 0]]
    p0Num = np.zeros(4)
    p0Denom = 0.0
    for i in range(len(dataSet)):
        p0Num += dataSet[i]
        p0Denom += sum(dataSet[i])
        print(dataSet[i],": aaaa :")
    print(p0Num)
    print(p0Denom)
    print(p0Num/p0Denom)

if __name__ == '__main__':
   test()