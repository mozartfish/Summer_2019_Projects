# this program explores writing a simple regression algorithm
# author: Pranav Rajan
# Date: June 10, 2019

from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')

# generate some random scattered data
xs = np.array([1, 2, 3, 4, 5, 6], dtype=np.float64)
ys = np.array([5, 4, 6, 5, 6, 7], dtype=np.float64)

# .scatter(xs, ys)
# plt.show()

def best_fit_slope(_and_interceptxs, ys):
    m = (((mean(xs) * mean(ys)) - mean(xs * ys)) /
         ((mean(xs) * mean(xs)) - mean(xs * xs)))

    b= mean(ys) - m * mean(xs)

    return m, b

m, b = best_fit_slope(xs, ys)

regression_line = [(m * x) + b for x in xs]

# predict some future data points
predict_x = 7

predict_y = (m * predict_x) + b
print(predict_y)

predict_x = 7
predict_y = (m*predict_x)+b

plt.scatter(xs,ys,color='#003F72',label='data')
plt.plot(xs, regression_line, label='regression line')
plt.legend(loc=4)
plt.show()
# print(m, b)
