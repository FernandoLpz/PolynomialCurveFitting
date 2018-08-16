# Example code for Polynomial Curve Fitting
#
# The present code shows how to apply the concept of curve fitting through
# a Polynomial of different degrees.
#
# In this case we are trying to fit the curve generated for the function
# sin(2*pi*x) ,where "x" are values in the range [-1,1] random numbers under
# a normal distribution.
#
# Author: Fernando Lopez-Velasco
#

# Import libraries
import numpy as np
import math as mt
import matplotlib.pyplot as plt

# Set the number of elements for the "x" vector
N =  1000
# Set the parameter of penalization
penalization = 0.1

# Function to generate the respecitve vector x and y.
def Generate_data():
    # Generating "x" from a random varible under a normal
    x = np.random.random_sample((N,))
    # Generating "y" from the function 2 * pi * x
    y_true = [(mt.sin(2 * mt.pi * i)) for i in x]

    return x, y_true

# Function to generate the Polynomial expansion according to the degree
def Polynomial(grade, x):
    # Generate random values for "w"
    w = np.random.randn(grade+1)

    # Intialize values to for loop
    y_hat = []
    y_aux = 0
    power = 0

    for x_element in x:
        for w_element in w:
            # y_aux = w_0x^0 + w_1x^1 + w_2x^2 + ... + w_Mx^M
            # where M is the degree of the polynomial
            y_aux += mt.pow(x_element, power) * w_element
            power += 1
        # save the sum for each element of vector x
        y_hat.append(y_aux)
        y_aux = 0
        power = 0

    return y_hat, w

# Calculate the error function based on least squares
def Error_Function_Least_Squares(y_true, y_hat, w):
    y_true = np.asarray(y_true)
    y_hat = np.asarray(y_hat)

    # set the regularizer
    regularizer = (penalization / 2) * (np.dot(w.T,w))

    return (1/2) * np.sum(np.power(np.add(y_hat, (-1) * y_true), 2)) + regularizer

# Calculate the error function based on RMS
def Error_Funcion_RMS(error_Least_Squares):
    return mt.sqrt((2 * error_Least_Squares)/ N)

# Gets data
x, y_true = Generate_data()

# Try with different polynomial degrees
for i in range(10):
    y_hat, w = Polynomial(i+1, x)
    error_Least_Squares = Error_Function_Least_Squares(y_true, y_hat, w)
    error_RMS = Error_Funcion_RMS(error_Least_Squares)

    # Print least squares error
    print("Polynomial Error LS, Grade ", i+1," :", error_Least_Squares)
    # Print RMS error
    print("Polynomial Error RMS, Grade ", i+1," :", error_RMS)
    print("\n")

    # Plot for every polynomial
    plt.plot(x,y_true,'ro')
    plt.plot(x,y_hat, 'b')
    plt.show()
