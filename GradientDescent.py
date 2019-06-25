# This program will be performing gradient descent on the function f(x) = x^4 - 3x^3 + 2
# Author: Pranav Rajan
# Date: May 31, 2019

current_location = 6 # The current x value that we are at
learning_rate = 0.01 # the learning rate determines the step
precision = 0.00001# The precision is where we determine where we want the algortihm to stop since we are satisfied with the how precise the answer is for our problem
num_iterations = 10000 # the number of iterations we will take to find a local min for the function

# The derivative of the function
df = lambda x : 4 * x**3 - 9 * x ** 2

for i in range(num_iterations):
    next_location = current_location
    next_location = current_location - learning_rate * df(current_location)
    step = next_location - current_location
    if abs(step) <= precision:
        break

print("A local minimum of the function is at: ", next_location)

