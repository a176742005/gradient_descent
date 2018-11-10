# -*- coding: utf-8 -*-

"""
    案例：利用NumPy实现梯度下降算法预测疾病
    任务：根据体重指数(BMI)和疾病发展的定量测量值(Y)使用梯度下降算法拟合出一条直线 y_hat = aX+b
"""
import numpy as np
import matplotlib.pylab as plt

data_path = './data/diabetes.csv'


def load_data(data_file):
    """
        功能：读取数据文件，加载数据
        参数：
            - data_file：文件路径
        返回：
            - data_arr：数据的多维数组表示
    """
    data_arr = np.loadtxt(data_file, delimiter=',', skiprows=1)
    return data_arr


def normalization(x, a, b):
    """
        对数据进行归一化操作
    """
    x = (x - b) / (a - b)
    return x


def get_gradient(theta, x, y):
    m = x.shape[0]
    y_estimate = x.dot(theta)
    error = y_estimate - y
    grad = 1.0/m * error.dot(x)
    cost = 1.0/(2 * m) * np.sum(error ** 2)
    return grad, cost


def gradient_descent(x, y, max_iter=1500, alpha=0.1):
    theta = np.random.randn(2)

    # 收敛阈值
    tolerance = 1e-6

    # 计数器
    iterations = 1

    is_converged = False
    while not is_converged:
        grad, cost = get_gradient(theta, x, y)
        new_theta = theta - alpha * grad

        # Print cost
        print('第{}次迭代，损失值 {:.4f}'.format(iterations, cost))

        # Stopping Condition
        if np.sum(abs(new_theta - theta)) < tolerance:
            is_converged = True
            print('参数收敛！！！')
            print('theta的值为：{}'.format(theta))

        if iterations >= max_iter:
            is_converged = True
            print('已至最大迭代次数{}'.format(max_iter))
            print('theta的值为：{}'.format(theta))

        iterations += 1
        theta = new_theta

    return theta


def show(x1, theta, y):
    # 绘制结果
    y_pred = theta[0] + theta[1] * x1[:, 1]
    plt.figure()

    # 绘制样本点
    plt.scatter(x1[:, 1], y)

    # 绘制拟合线
    plt.plot(x1[:, 1], y_pred, c='red')
    plt.show()


def main():
    """
        主函数
    """
    data_arr = load_data(data_path)
    x = data_arr[:, 0].reshape(-1, 1)
    x = normalization(x, np.max(x), np.min(x))
    y = data_arr[:, 1]
    y = normalization(y, np.max(y), np.min(y))
    x1 = np.hstack((np.ones_like(x), x))
    theta = gradient_descent(x1, y, alpha=0.1, max_iter=10000)
    show(x1, theta, y)


if __name__ == '__main__':
    main()


















