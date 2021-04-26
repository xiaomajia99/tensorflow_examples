import numpy as np


def getRandomFloat(low : float = 0.0, high : float = 10.0, point_num : int = 5) -> float:
    """
    获取一个指定范围的随机float数   范围  ： [low, high)
    
    low : 最小值
    high : 最大值
    point_num : 保留几位小数

    return 一个浮点数，None说明参数不对
    """
    if low >= high or point_num < 0:
        return None
    else:
        return round(np.random.uniform(low, high), point_num)


if __name__ == "__main__":
    getRandomFloat()
    