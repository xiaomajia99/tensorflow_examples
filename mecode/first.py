
import tensorflow as tf
import logging
import numpy as np


def helloWord():
    hw = tf.constant("Hello World!!!")
    logging.info(hw.numpy())

def baseCal(x : int, y : int):
    x_c = tf.constant(x)
    y_c = tf.constant(y)
    logging.debug("x : %f, y : %f, x + y : %f, x - y : %f, x * y : %f, x / y : %f",
                    x_c.numpy(), 
                    y_c.numpy(), 
                    tf.add(x_c, y_c).numpy(), 
                    tf.subtract(x_c, y_c).numpy(), 
                    tf.multiply(x_c, y_c).numpy(), 
                    tf.divide(x_c,y_c).numpy())
    
def collectionCal(a):
    a_c = tf.constant(a)
    logging.debug("a : %s, a_mean : %f, a_sum : %f", 
                    a_c.numpy(),
                    tf.reduce_mean(a_c).numpy(), 
                    tf.reduce_sum(a_c).numpy())

def matrixCal(a, b):
    a_c = tf.constant(a)
    b_c = tf.constant(b)
    c = tf.matmul(a_c, b_c)

    logging.debug("a : %s, b : %s, a * b : %s", 
                    str(a_c.numpy()), 
                    str(b_c.numpy()), 
                    str(c.numpy()))
    

if __name__ == "__main__":
    print("test")