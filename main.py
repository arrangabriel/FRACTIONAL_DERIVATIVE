from sympy import *
from manimlib import *
import numpy as np
import matplotlib.pyplot as plt

# TODO create input parser using sympy
def parse_function(input_string):
    pass

# takes a single derivative of a function
def derivative(function, variable):
    function_prime = function.diff(variable)
    return function_prime

# gets the 'weighted average' of two functions at an x value
def weighted_function_average(function_1, function_2, point, weight):
    function_1_val = function_1(point)
    function_2_val = function_2(point)
    delta_weighted = np.abs(function_1_val - function_2_val) * weight
    if function_1_val < function_2_val:
        return function_1_val + delta_weighted
    else:
        return function_1_val - delta_weighted

# calculate points for all x_values
def calculate_points(function_main_lambda, function_prime_1_lambda, function_prime_2_lambda, x_values, weight):
    points = {i: [function_main_lambda(i),
                  function_prime_1_lambda(i),
                  function_prime_2_lambda(i),
                  weighted_function_average(function_prime_1_lambda, function_prime_2_lambda, i, weight)]
              for i in x_values}
    return points

# differentiate upper and lower bounds
def endpoint_derivatives(function_1, function_2, lower_derivative_magnitude, upper_derivative_magnitude, variable):
    for i in range(lower_derivative_magnitude):
        function_1 = derivative(function_1, variable)
    for i in range(upper_derivative_magnitude):
        function_2 = derivative(function_2, variable)
    return function_1, function_2

# for calculating a single fractional derivative on an interval
def fractional_derivative_interval(function, variable, magnitude, interval, npoints):
    if magnitude < 0:
        raise Exception("Cannot take negative derivative. Try an integral or something.")

    function_1 = function
    function_2 = function

    # lower and upper derivative magnitudes
    lower_derivative_magnitude = int(np.floor(magnitude))
    upper_derivative_magnitude = int(np.ceil(magnitude))

    # trailing digits of the magnitude varible
    weight = magnitude - lower_derivative_magnitude

    # divide interval into n x-values
    x_values = np.linspace(interval[0], interval[1], npoints)

    # differentiate to upper and lower derivative
    function_prime_1, function_prime_2 = endpoint_derivatives(function_1,
                                                              function_2,
                                                              lower_derivative_magnitude,
                                                              upper_derivative_magnitude,
                                                              variable)

    # lambdify functions
    function_lambda = lambdify(variable, function)
    function_prime_1_lambda = lambdify(variable, function_prime_1)
    function_prime_2_lambda = lambdify(variable, function_prime_2)

    # calculate the fractional derivative for points in the generated interval
    points = calculate_points(function_lambda, function_prime_1_lambda, function_prime_2_lambda, x_values, weight)

    return points

# for doing n equally spaced derivatives between magnitude and magnitude + 1
def fractional_derivatives(function, variable, lower_derivative, n, interval, npoints):
    if lower_derivative < 0:
        raise Exception("Cannot take negative derivatives. Try an integral or something.")

    function_1 = function
    function_2 = function

    # lower and upper derivative magnitudes
    lower_derivative_magnitude = lower_derivative
    upper_derivative_magnitude = lower_derivative + 1

    # divide interval into n x-values
    x_values = np.linspace(interval[0], interval[1], npoints)

    # divide derivative magnitudes into n values
    derivative_magnitudes = np.linspace(0, 1, n)

    # differentiate to upper and lower derivative
    function_prime_1, function_prime_2 = endpoint_derivatives(function_1,
                                                              function_2,
                                                              lower_derivative_magnitude,
                                                              upper_derivative_magnitude,
                                                              variable)

    # lambdify functions
    function_lambda = lambdify(variable, function)
    function_prime_1_lambda = lambdify(variable, function_prime_1)
    function_prime_2_lambda = lambdify(variable, function_prime_2)

    # calculate points
    points = {
        magnitude + lower_derivative_magnitude: calculate_points(function_lambda,
                                                                 function_prime_1_lambda,
                                                                 function_prime_2_lambda,
                                                                 x_values,
                                                                 magnitude)
        for magnitude in derivative_magnitudes
    }

    return points

def main():
    x = Symbol("x")
    function_1 = x ** 3 + 1
    function_2 = sin(x)
    function_1_points = fractional_derivative_interval(function_1, x, 1.1, [0,3], 50)
    function_2_points = fractional_derivative_interval(function_2, x, 1.1, [0,3], 50)
    #print(fractional_derivatives(function_1, x, 1, 4, [0,3], 10))
    #plt.plot(*zip(*sorted(function_1_points.items())))
    plt.plot(*zip(*sorted(function_2_points.items())))
    plt.show()

if __name__ == '__main__':
    main()

# this is pretty bad
class SineExample(Scene):
    def construct(self):
        axes = Axes(
            x_range=(-6, 6),
            y_range=(-6, 6),
            height=12,
            width=12,
            axis_config={
                "stroke_color": GREY_A,
                "stroke_width": 2,
                "include_tip": False
            },
        )

        self.add(axes)

        x = Symbol("x")
        function = sin(x) + x**2
        points = fractional_derivative_interval(function, x, 1.3, [-3,3], 300)
        colors = [RED, GREEN, BLUE, PURPLE]

        for x, y_values in points.items():
            for i in range(len(y_values)):
                self.add(Dot(color=colors[i]).move_to(axes.c2p(x, y_values[i])))
        self.wait()