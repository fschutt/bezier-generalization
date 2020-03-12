import matplotlib
from math import floor, ceil
import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid.axislines import SubplotZero

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

def get_sum_x(points, power):
    s = 0
    for p in points:
        s += pow(p.x, power)
    return s

def get_sum_y(points, power):
    s = 0
    for p in points:
        s += pow(p.x, power)
    return s

# sum of x^2*y
def get_sum_x_2y(points):
    s = 0
    for p in points:
        s += pow(p.x, 2) * p.y
    return s

# sum of x*y
def get_sum_x_y(points):
    s = 0
    for p in points:
        s += p.x * p.y
    return s

# sum of y
def get_sum_y(points):
    s = 0
    for p in points:
        s += p.y
    return s

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

    curve_begin = floor(curve_begin.x)
    curve_end = ceil(curve_end.x)

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
    curve_quadratic  = try_fit_curve_cubic(points)
    return curve_quadratic

    #curve_cubic = try_fit_curve_cubic(points)
    #
    #if (calc_error_cubic(points, curve_cubic) > calc_error_quadratic(points, curve_quadratic)):
    #    return curve_cubic
    #else:
    #    return curve_quadratic

def try_fit_curve_quadratic(points):

    if len(points) < 3:
        raise "Too few points"

    s00 = len(points)                       # sum of x^0 * y^0  ie 1 * number of entries
    s10 = get_sum_x(points, 1)              # sum of x
    s20 = get_sum_x(points, 2)              # sum of x^2
    s30 = get_sum_x(points, 3)              # sum of x^3
    s40 = get_sum_x(points, 4)              # sum of x^4

    s21 = get_sum_x_2y(points)   # sum of x^2*y
    s11 = get_sum_x_y(points)    # sum of x*y
    s01 = get_sum_y(points)      # sum of y

    D = (s40 * (s20 * s00 - s10 * s10) - \
         s30 * (s30 * s00 - s10 * s20) + \
         s20 * (s30 * s10 - s20 * s20))

    Da = (s21 * (s20 * s00 - s10 * s10) - \
          s11 *(s30 * s00 - s10 * s20) + \
          s01 *(s30 * s10 - s20 * s20))

    Db = (s40 * (s11 * s00 - s01 * s10) - \
          s30 * (s21 * s00 - s01 * s20) + \
          s20 * (s21 * s10 - s11 * s20))

    Dc = (s40 * (s20 * s01 - s10 * s11) - \
          s30 * (s30 * s01 - s10 * s21) + \
          s20 * (s30 * s11 - s20 * s21))

    return [Da/D, Db/D, Dc/D]

def try_fit_curve_cubic(points):
    pass

def calc_error_cubic(points, curve):
    pass

def calc_error_quadratic(points, curve):
    pass

def main():

    points = [
        Point(-1.74, 3.03),
        Point(-1.02, 1.04),
        Point(1.18, 1.39),
        Point(1.44, 4.0),
        Point(1.56, 3.04),
        Point(0.58, 0.66),
        Point(0.46, -0.32),
        Point(-0.94, 0.28),
        Point(-1.2, 2.22),
    ]

    # f(x) = 1x² + 0x + 0
    curve = try_fit_curve_quadratic(points)
    
    print(curve)
    # f(x) = 1x³ + 0x² + 0x + 0
    # curve2 = [1, 0, 0, 0]

    open_window_and_show_results(points, curve)

main()