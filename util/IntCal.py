
import numpy as np

def getRandomInt(low : int = 0, high : int = 10) -> int:
    """
    获取一个随机值，范围 [low, high)

    low : 最小值    默认0
    high ： 最大值  默认10

    return : 一个随机int值
    """
    if low >= high:
        return None
    else:
        return np.random.randint(low=low, high=high)

def getRandomIntList(low : int = 0, high : int = 10, length : int = 10) -> list:
    """
    获取一个有指定范围的产生的推荐值构建的列表   范围 ： [low, high)

    low : 最小值
    high : 最大值
    length : 列表长度

    return ： 一个由随机int的列表
    """
    ret = []
    if low >= high or length <= 0:
        return ret
    else:
        for i in range(0, length):
            ret.append(getRandomInt(low, high))
        return ret

def getRandomIntMatrix(low : int = 0, high : int = 10, shape : tuple = (2,2)) -> list:
    """
    获取一个指定范围和维度的矩阵，   范围 ：[low, high)
    
    low : 最小值
    high : 最大值
    shape : 矩阵大小

    return : 一个随机int的矩阵
    """
    ret = []
    if low >= high or shape[0] < 1 or shape[1] < 1:
        return ret

    for i in range(0, shape[0]):
        element = []
        for j in range(0, shape[1]):
            element.append(getRandomInt(low, high))
        ret.append(element)
    return ret

if __name__ == "__main__":
    print(getRandomIntMatrix())