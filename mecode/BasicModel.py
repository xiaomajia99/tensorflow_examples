import tensorflow as tf
import logging

class LinearRegression:
    """
    一元线性回归模型， 初始化init模型的的时候，要有训练数据和模型参数
    train_x : 自变量
    train_y : 因变量
    train_w : 权重
    train_b ：偏执
    learn_rate : 学习率
    train_steps : 训练次数
    """
    
    def __init__(self, 
                train_x : list,
                train_y : list,
                train_w : float = 1.0,
                train_b : float = 1.0,
                learn_rate : float = 0.01, 
                train_steps : int = 100):
        self.X = train_x
        self.Y = train_y
        self.W = tf.keras.backend.variable(train_w, name="weight")
        self.B = tf.keras.backend.variable(train_b, name="bias")
        self.learn_rate = learn_rate
        self.train_steps = train_steps

    def linearRegression(self):
        return self.W * self.X + self.B

    def meanSquare(self, y_pred, y_true):
        return tf.keras.backend.mean(tf.keras.backend.square(y_pred - y_true))

    def fit(self):
        opt = tf.keras.optimizers.SGD(learning_rate = self.learn_rate)

        for step in range(0, self.train_steps):
            self.run(opt)
            if step % 10 == 0:
                logging.debug("w : %f, b : %f", self.W.numpy(), self.B.numpy())
        
    def run(self, opt):
        with tf.GradientTape() as gt:
            y_pred = self.linearRegression()
            loss = self.meanSquare(y_pred, self.Y)

        gradients = gt.gradient(loss, [self.W, self.B])
        opt.apply_gradients(zip(gradients, [self.W, self.B]))

class LogisticRegression:
    def __init__(self):
        self.num = 1