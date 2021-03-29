from manim import *
from sympy import *
from manimlib import *
import numpy as np
import matplotlib.pyplot as plt

# TODO create input parser using sympy
def parse_function(input_string):
    pass

# TODO refactor the multiple derivatives function to an object
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
def endpoint_derivatives(function, lower_derivative_magnitude, upper_derivative_magnitude, variable):
    function_1, function_2 = function, function
    for i in range(lower_derivative_magnitude):
        function_1 = derivative(function_1, variable)
    for i in range(upper_derivative_magnitude):
        function_2 = derivative(function_2, variable)
    return function_1, function_2

# for doing n equally spaced derivatives between magnitude and magnitude + 1
def fractional_derivatives(function, variable, lower_derivative, n, interval, npoints):
    if lower_derivative < 0:
        raise Exception("Cannot take negative derivatives. Try an integral or something.")

    # lower and upper derivative magnitudes
    lower_derivative_magnitude = lower_derivative
    upper_derivative_magnitude = lower_derivative + 1

    # divide interval into n x-values
    x_values = np.linspace(interval[0], interval[1], npoints)

    # divide derivative magnitudes into n values
    derivative_magnitudes = np.linspace(0, 1, n)

    # differentiate to upper and lower derivative
    function_prime_1, function_prime_2 = endpoint_derivatives(function,
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

class Fractional_Derivative:
    def __init__(self, function, variable, lower_derivative_magnitude: int):
        if lower_derivative_magnitude < 0:
            raise Exception("Cannot take negative derivative. Try an integral or something.")
        self.base_function = function
        self.variable = variable
        self.base_function_lambda = lambdify(self.variable, function)
        self.lower_magnitude = lower_derivative_magnitude
        self.lower_derivative, self.upper_derivative = endpoint_derivatives(self.base_function,
                                                                  self.lower_magnitude,
                                                                  self.lower_magnitude + 1,
                                                                  self.variable)
        self.lower_derivative_lambda = lambdify(self.variable, self.lower_derivative)
        self.upper_derivative_lambda = lambdify(self.variable, self.upper_derivative)

    def eval_function(self, x: float):
        return self.base_function_lambda(x)

    def eval_lower_derivative(self, x: float):
        return self.lower_derivative_lambda(x)

    def eval_upper_derivative(self, x: float):
        return self.upper_derivative_lambda(x)

    def eval_fractional_derivative(self, x: float, magnitude):
        if magnitude < 0 or magnitude > 1:
            raise Exception("Magnitude must be between o and 1 inclusive.")
        lower_value = self.lower_derivative_lambda(x)
        upper_value = self.upper_derivative_lambda(x)
        delta_weighted = np.abs(lower_value - upper_value) * magnitude
        if lower_value < upper_value:
            return lower_value + delta_weighted
        else:
            return lower_value - delta_weighted

    def eval_function_list(self, x_values: [float]):
        return [self.eval_function(x) for x in x_values]

    def eval_lower_derivative_list(self, x_values: [float]):
        return [self.lower_derivative_lambda(x) for x in x_values]

    def eval_upper_derivative_list(self, x_values: [float]):
        return [self.upper_derivative_lambda(x) for x in x_values]

    def eval_fractional_derivative_list(self, lower_values: [float], upper_values: [float], magnitude):
        if magnitude < 0 or magnitude > 1:
            raise Exception("Magnitude must be between o and 1 inclusive.")
        fractional_values = []
        for i in range(len(lower_values)):
            lower_value = lower_values[i]
            upper_value = upper_values[i]
            delta_weighted = np.abs(lower_value - upper_value) * magnitude
            if lower_value < upper_value:
                fractional_values.append(lower_value + delta_weighted)
            else:
                fractional_values.append(lower_value - delta_weighted)
        return fractional_values

def main():
    x = Symbol("x")
    function = sin(x)

    function_1_instance = Fractional_Derivative(function, x, 1)
    x_values = np.linspace(0, np.pi, 20)
    magnitude = 0.4

    function_values = function_1_instance.eval_function_list(x_values)
    lower_derivative_values = function_1_instance.eval_lower_derivative_list(x_values)
    upper_derivative_values = function_1_instance.eval_upper_derivative_list(x_values)
    fractional_derivative_values = function_1_instance.eval_fractional_derivative_list(lower_derivative_values, upper_derivative_values, magnitude)
    plt.plot(x_values, function_values, label= function_1_instance.base_function)
    plt.plot(x_values, lower_derivative_values, label= f"{function_1_instance.lower_magnitude} - derivative")
    plt.plot(x_values, upper_derivative_values, label= f"{function_1_instance.lower_magnitude + 1} - derivative")
    plt.plot(x_values, fractional_derivative_values, label= f"{magnitude + function_1_instance.lower_magnitude} - derivative.")
    plt.legend(loc= "lower right")
    plt.show()

if __name__ == '__main__':
    main()

# manim
class SmoothGraphFromSetPoints(VMobject):
    def __init__(self, set_of_points, **kwargs):
        super().__init__(**kwargs)
        self.set_points_smoothly(set_of_points)

class SineExample(Scene):
    def construct(self):
        axes = Axes((-3,3), (-1,3))
        axes.add_coordinate_labels()
        self.play(Write(axes, lag_ratio=0.01, run_time=1))

        x = Symbol("x")
        function = sin(x)

        function_1_instance = Fractional_Derivative(function, x, 1)
        x_values = np.linspace(-np.pi, np.pi, 20)
        magnitude = 0.4

        function_values = function_1_instance.eval_function_list(x_values)
        points_1 = [axes.coords_to_point(x, y) for x, y in [*zip(x_values, function_values)]]
        graph_1 = SmoothGraphFromSetPoints(points_1, color=ORANGE)

        lower_derivative_values = function_1_instance.eval_lower_derivative_list(x_values)
        points_2 = [axes.coords_to_point(x, y) for x, y in [*zip(x_values, lower_derivative_values)]]
        graph_2 = SmoothGraphFromSetPoints(points_2, color=BLUE)

        upper_derivative_values = function_1_instance.eval_upper_derivative_list(x_values)
        points_3 = [axes.coords_to_point(x, y) for x, y in [*zip(x_values, upper_derivative_values)]]
        graph_3 = SmoothGraphFromSetPoints(points_3, color=GREEN)

        fractional_derivative_values = function_1_instance.eval_fractional_derivative_list(lower_derivative_values, upper_derivative_values, magnitude)
        points_4 = [axes.coords_to_point(x, y) for x, y in [*zip(x_values, fractional_derivative_values)]]
        graph_4 = SmoothGraphFromSetPoints(points_4, color=RED)

        self.play(ShowCreation(graph_1, run_time=2))
        self.wait(0.5)
        self.play(ShowCreation(graph_2, run_time=2))
        self.wait(0.5)
        self.play(ShowCreation(graph_3, run_time=2))
        self.wait(0.5)
        self.play(ShowCreation(graph_4, run_time=2))
