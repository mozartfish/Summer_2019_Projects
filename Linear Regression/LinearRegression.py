# Radio example from ml-cheat sheet
# Author: Pranav Rajan
# Date: May 30, 2019

import pandas as pd


# import data


def get_data(file_name):
    data = pd.read_csv('Radio.csv')
    # print(data)
    companies = []
    radio = []
    sales = []
    for company, radios, Sales in zip(data['Company'], data['Radio ($)'], data['Sales']):
        companies.append(company)
        radio.append(radios)
        sales.append(sales)

    return companies, radio, sales

companies, radio, sales = get_data('Radio.csv')

def predict_sales(radio, weight, bias):
    return weight * radio + bias # a linear equation of the form y = mx + b

# cost function. There are many kinds of cost functions and the one used here is MSE (mean square error)
def cost_function(radio, sales, weight, bias):
    companies = len(radio)
    total_error = 0.0
    for i in range(companies):
        total_error +=(sales[i] - (weight * radio[i] + bias)) ** 2
        return total_error / companies

# compute gradient for the cost function. The resulting gradient tells us the slope of the cost function and the direction we take to reduce the cost function
# Learning Rate - the size of the updates (steps) in the direction of the negative to get to a relative minima of some function
# With a high learning rate we can travel a greater distance per step but risk overshooting the local min since the slope is changing
# With a low learning rate, we can move in the correct directon and step for the gradient since we recalculate it frequently but it is very costly in terms of time

def update_weights_biases(radio, sales, weight, bias, learning_rate):
    weight_deriv = 0;
    bias_deriv = 0;
    companies = len(radio)

    for i in range(companies):
        # Calculate weight partial derivatives
        # -2x(y - (mx + b)
        weight_deriv += -2 * radio[i] * (sales[i] - (weight * radio[i] * bias))

        # Calculate the bias partial derivatives
        # -2(y - (mx + b))
        bias_deriv += -2 * (sales[i] - (weight * radio[i] + bias))

        # Subtract the derivatives because we want the negative gradient
        weight -= (weight_deriv / companies) * learning_rate
        bias -= (bias_deriv / companies) * learning_rate

    return weight, bias

# Training
# Training the model is iteratively improving the prediction equation by looping through the data set multiple times
# each time updating the weight and bias values in the direction indicated by the negative gradient
# Once we reach an acceptable error threshold, training is done or when subsequent training iterations fail to reduce the cost

# Training steps:
# 1) Initialize the weights (some randomly generated values)
# 2) Set the hyperparameters(the learning rate and number of iterations)

def train(radio, sales, weight, bias, learning_rate, iterations):
    cost_history = []

    for j in range(iterations):
        weight, bias = update_weights_biases(radio, sales, weight, bias, learning_rate)

        # Calculate the cost for training samples
        cost = cost_function(radio, sales, weight, bias)
        cost_history.append(cost)

        # Training Progress

        if j % 10 == 0:
            print("iter = {:d}    weight = {:.2f}    bias = {:.4f}    cost = {:.2f}".format(j, weight, bias, cost))

    return weight, bias, cost_history

