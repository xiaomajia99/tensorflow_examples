
import logging
import os 
import sys


rootPath = os.getcwd() + "/tensorflow_examples"
logPath = rootPath + "/logs/"
dataPath = rootPath + "/data/"
codePath = rootPath + "/mecode/"
utilPath = rootPath + "/util/"

sys.path.append(codePath)
sys.path.append(utilPath)

import first, BasicModel
import IntCal

print(rootPath, logPath)

def configLog(fileName = "logger.log"):
    """
    配置logging日志
    """
    filePath = logPath + fileName
    logging.basicConfig(level=logging.DEBUG, filename=filePath, format="%(asctime)s %(filename)s %(levelname)s %(message)s")
    

def firstMain():
    configLog()
    #first.helloWord()
    #first.baseCal(IntCal.getRandomInt(), IntCal.getRandomInt())
    #a = IntCal.getRandomIntList()
    #first.collectionCal(a)
    #a = IntCal.getRandomIntMatrix()
    first.matrixCal(IntCal.getRandomIntMatrix(), IntCal.getRandomIntMatrix())
    
def basicModelMain():
    configLog()
    m1 = BasicModel.LinearRegression(train_x = IntCal.getRandomIntList(length=20),train_y = IntCal.getRandomIntList(length=20))
    m1.fit()

    #m2 = BasicModel.LogisticRegression()
    #logging.debug(m1.val)
    #logging.debug(m2.num)
    

def main():
    #firstMain()
    basicModelMain()

if __name__ == "__main__":
    main()