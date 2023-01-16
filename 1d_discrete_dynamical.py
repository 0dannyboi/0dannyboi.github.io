'''
I wrote this code to analyze the behavior of discrete dynamical systems of a
single variable and a single parameter. The logistic map,
f(x) = a * x * (1 - x), is a famous example. It is a function of the variable
x and a single parameter a. The accompanying comments will refer
to an input map as 'f(x)', an input variable as 'x', and an input parameter
as 'a' for simplicity. The set of values containing the initial value
and subsequent values over a series of iterations of the map
is referred to as the orbit.

As input, it takes a string containing the expression governing the map, as
well as a string representing the variable and a string representing the
parameter. It can draw cobweb plots for different initial conditions as well
as a bifurcation diagram.


It can also return fixed points (provided the symbolic engine can solve them),
return the derivative of the map, and evaluate the map at a given set of
conditions.

Daniel Foster, May 2021

Additional References:
https://mathworld.wolfram.com/WebDiagram.html
https://mathworld.wolfram.com/Bifurcation.html
http://csc.ucdavis.edu/~chaos/courses/poci/Readings/ch2.pdf
'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from sympy import*


class discrete_map:

    def __init__(self, expression, variable, parameter):
        self.parameter = symbols(parameter)
        self.variable = symbols(variable)
        self.expr = sympify(expression)
        self.derivative = diff(self.expr, self.variable)

    def evaluate(self, variable_value, parameter_value):
        return self.expr.subs({self.variable: variable_value,
                               self.parameter: parameter_value})

    def fixed_points(self):
        try:
            f_points = solve(self.expr - self.variable, self.variable)
            # Calls on the symbolic solver to find fixed points, or values
            # of the variable which return the same value under mapping.
            # This is achieved by finding the zeros of 'f(x) - x'.
        except:
            print("Error finding fixed points analytically.")
            f_points = False
        return f_points

    def cobweb_animation(self, a, n, x_i,save=False):
        # this method returns the animation of the cobweb plot for the map of
        # parameter a, with the variable starting at 'x_i' over 'n' iterations.
        n_fil = 3  # controls the number of steps for plotting

        def create_steps(x_temp, y_temp):
            # turns values of each iteration into plottable format
            x_new = []
            y_new = []
            for a in range(0, len(x_temp) - 1):
                delta_x = x_temp[a + 1] - x_temp[a]
                delta_y = y_temp[a + 1] - y_temp[a]
                dx = delta_x / n_fil
                dy = delta_y / n_fil
                for b in range(0, n_fil):
                    x_new.append(float(x_temp[a] + b * dx))
                    y_new.append(float(y_temp[a] + b * dy))
            return x_new, y_new

        def update(i):  # updates plotting animation
            if (i == 0):
                ax.clear()
            plt.title("Cobweb plot of " + "$" + str(latex(self.expr)) + "$")
            plt.xlabel("x")
            plt.ylabel("y")
            plt.plot(x_s, y_s, c="blue")  # plots the map on the domain x_s
            plt.plot(x_s, x_s, c="black")  # plots the line "y = x"
            plt.plot(x_1[i:i+2], y_1[i:i+2], color=colors[i])
            # plots a single iteration
            plt.xlim(0, 2)
            plt.ylim(0, 2)
            # the x and y limits are entered manually
            return fig,

        x_s = np.linspace(0, 2, 1000)  # variable values
        y_s = [self.evaluate(b, a) for b in x_s]   # map values on domain x_s
        colors = plt.cm.jet(np.linspace(0, 1, 2 * n))
        fig, ax = plt.subplots()  # initialize plotting
        x_d = [x_i]  # contains values of map for the orbit starting at "x_i"
        y_d = [0]  # starts plotting from the "x-axis"
        for i in range(0, n):  # steps through each iteration of the map
            x1 = x_d[-1]
            y1 = self.evaluate(x1, a)
            x_d.append(x1)
            y_d.append(y1)
            x_d.append(y1)
            y_d.append(y1)
        x_1, y_1 = create_steps(x_d, y_d)
        ani = animation.FuncAnimation(fig, update, frames=(n_fil - 1) *
                                      (n - 1), interval=10)
        if save:
            ani.save(save, writer=animation.PillowWriter(fps=30))
        plt.show()
        return ani

    def bifurcate(self, parameter_min, parameter_max, n_iter, final_n, var_i):
        parameter_range = np.linspace(parameter_min, parameter_max, 200)
        for p in parameter_range:
            param_axis = np.zeros(final_n) + p
            final_vals = []  # list to contain the final values of the orbit
            var = var_i
            for itera in range(n_iter):
                var = float(self.evaluate(var, p))
                if (n_iter - itera <= final_n):
                    final_vals.append(var)
            plt.scatter(param_axis, final_vals, c="red", s=0.03)
            # produces a scatter plot along the slice p, with the "y" values
            # being the final values of the orbit
        plt.title("Bifurcation diagram for " + str(self.expr))
        plt.xlabel(str(self.parameter))
        plt.ylabel("Final " + str(final_n) + " values of " +
                   str(self.variable))
        plt.show()
    
        
