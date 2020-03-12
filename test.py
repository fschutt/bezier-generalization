__author__ = 'Dania'
import matplotlib
matplotlib.use('TkAgg')
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from tkinter import *
from mpl_toolkits.axes_grid.axislines import SubplotZero


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

# Returns the points for f(x) = ax² + bx + c in the form of a [Point]
def generate_curve_points_quadratic(lim_range, a, b, c):
    values = []
    
    for x in lim_range:
        y =     a * x ** 2      \
            +   b * x ** 1      \
            +   c * x ** 0
        values.append(Point(x, y))

    return values

# Returns the points for f(x) = ax³ + bx² + cx + d in the form of a [Point]
def generate_curve_points_cubic(lim_range, a, b, c, d):
    values = []
    
    for x in lim_range:
        y =     a * x ** 3      \
            +   b * x ** 2      \
            +   c * x ** 1      \
            +   d * x ** 0
        values.append(Point(x, y))

    return values

# Shows the plot of points + curve
def open_window_and_show_results(points, curve):

    # Generate the curve
    curve_begin = min(points, key=lambda p: p.y)
    curve_end = max(points, key=lambda p: p.x)

    curve_begin = curve_begin.x
    curve_end = curve_end.x

    fig = plt.figure(1)
    ax = SubplotZero(fig, 111)
    fig.add_subplot(ax)

    for direction in ["xzero", "yzero"]:
        ax.axis[direction].set_axisline_style("-|>")
        ax.axis[direction].set_visible(True)

    for direction in ["left", "right", "bottom", "top"]:
        ax.axis[direction].set_visible(False)


    curve_points = []

    if (len(curve) == 3):
        curve_points = generate_curve_points_quadratic([x * 0.1 for x in range(curve_begin * 10 , curve_end * 10)], curve[0], curve[1], curve[2]) # values for x²
    elif (len(curve) == 4):
        curve_points = generate_curve_points_cubic([x * 0.1 for x in range(curve_begin * 10 , curve_end * 10)], curve[0], curve[1], curve[2], curve[3]) # values for x²
    else:
        raise "Length of curve has to be 3 or 4"

    # Plot curve
    x_curve = []
    y_curve = []

    for p in curve_points:
        x_curve.append(p.x)
        y_curve.append(p.y)

    plt.plot(x_curve, y_curve, color='blue')


    # Plot points
    x_points = []
    y_points = []

    for p in points:
        x_points.append(p.x)
        y_points.append(p.y)

    plt.plot(x_points, y_points, 'ro')

    plt.show()

# Returns an array of either 3 or 4 values, depending 
# on which curve has the least error
def fit_curve(points): 
    curve_cubic = try_fit_curve_cubic(points)
    curve_quadratic  = try_fit_curve_cubic(points)

    if (calc_error_cubic(points, curve_cubic) > calc_error_quadratic(points, curve_quadratic)):
        return curve_cubic
    else:
        return curve_quadratic

def try_fit_curve_quadratic(points):
    pass

def try_fit_curve_cubic(points):
    pass

def calc_error_cubic(points, curve):
    pass

def calc_error_quadratic(points, curve):
    pass

def main():

    points = [
        Point(-4, 5),
        Point(6, 7)
    ]

    # f(x) = 1x² + 0x + 0
    curve = [1, 0, 0]
    # f(x) = 1x³ + 0x² + 0x + 0
    # curve2 = [1, 0, 0, 0]

    open_window_and_show_results(points, curve)

main()