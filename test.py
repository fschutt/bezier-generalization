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

# sum of x^3*y
def get_sum_x_3y(points):
    s = 0
    for p in points:
        s += pow(p.x, 3) * p.y
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

# Returns the point at t of a curve [p0, p1, p2, p3]
def cubic_bezier_curve(p0, p1, p2, p3, t):
    # B(t) = (1-t)**3 p0 + 3(1 - t)**2 t P1 + 3(1-t)t**2 P2 + t**3 P3
    x = (1-t)*(1-t)*(1-t)*p0.x + 3*(1-t)*(1-t)*t*p1.x + 3*(1-t)*t*t*p2.x + t*t*t*p3.x;
    y = (1-t)*(1-t)*(1-t)*p0.y + 3*(1-t)*(1-t)*t*p1.y + 3*(1-t)*t*t*p2.y + t*t*t*p3.y;
    return Point(x, y)

# Returns the points making up the bezier curve
def draw_bezier_curve(p0, p1, p2, p3):
    points = []

    for t in range(0, 100):
        t = t / 100.0
        points.append(cubic_bezier_curve(p0, p1, p2, p3, t))

    return points

# Shows the plot of points + curve
def open_window_and_show_results(points, curve, bezier_points):

    # Generate the curve
    curve_begin = min(points, key=lambda p: p.x)
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

    plot_points(plt, curve_points, 'blue')

    plot_points(plt, points, 'ro')

    plot_points(plt, bezier_points, 'go')

    plot_points(plt, draw_bezier_curve(bezier_points[0], bezier_points[1], bezier_points[2], bezier_points[3]), 'orange')
    
    plt.show()

def plot_points(plt, points, options):
    
    # Plot points
    x_points = []
    y_points = []

    for p in points:
        x_points.append(p.x)
        y_points.append(p.y)

    plt.plot(x_points, y_points, options)

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

    s00 = get_sum_x(points, 0) # len(points) # sum of x^0, ie 1 * number of entries
    s10 = get_sum_x(points, 1)               # sum of x^1
    s20 = get_sum_x(points, 2)               # sum of x^2
    s30 = get_sum_x(points, 3)               # sum of x^3
    s40 = get_sum_x(points, 4)               # sum of x^4

    s21 = get_sum_x_2y(points)   # sum of x^2*y
    s11 = get_sum_x_y(points)    # sum of x^1*y
    s01 = get_sum_y(points)      # sum of x^0y

    # [ S40  S30  S20 ] [ a ]   [ S21 ]
    # [ S30  S20  S10 ] [ b ] = [ S11 ]
    # [ S20  S10  S00 ] [ c ]   [ S01 ]

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


def det_4x4(matrix):
    mat0 = matrix[0][0] * det_3x3([
        [matrix[1][1], matrix[1][2], matrix[1][3]],
        [matrix[2][1], matrix[2][2], matrix[2][3]],
        [matrix[3][1], matrix[3][2], matrix[3][3]]
    ])

    mat1 = matrix[1][0] * det_3x3([
        [matrix[0][1], matrix[0][2], matrix[0][3]],
        [matrix[2][1], matrix[2][2], matrix[2][3]],
        [matrix[3][1], matrix[3][2], matrix[3][3]]
    ])

    mat2 = matrix[2][0] * det_3x3([
        [matrix[0][1], matrix[0][2], matrix[0][3]],
        [matrix[1][1], matrix[1][2], matrix[1][3]],
        [matrix[3][1], matrix[3][2], matrix[3][3]]
    ])

    mat3 = matrix[3][0] * det_3x3([
        [matrix[0][1], matrix[0][2], matrix[0][3]],
        [matrix[1][1], matrix[1][2], matrix[1][3]],
        [matrix[2][1], matrix[2][2], matrix[2][3]]
    ])

    return mat0 - mat1 + mat2 - mat3

# Returns the determinant of a 3x3 matrix
def det_3x3(matrix):

    mat0 = matrix[0][0] * det_2x2([
        [matrix[1][1], matrix[1][2]],
        [matrix[2][1], matrix[2][2]]
    ])

    mat1 = matrix[1][0] * det_2x2([
        [matrix[0][1], matrix[0][2]],
        [matrix[2][1], matrix[2][2]]
    ])
    
    mat2 = matrix[2][0] * det_2x2([
        [matrix[0][1], matrix[0][2]],
        [matrix[1][1], matrix[1][2]]
    ])

    return mat0 - mat1 + mat2

def det_2x2(matrix):
    return matrix[0][0] * matrix[1][1] - matrix[1][0] * matrix[0][1]

# def try_fit_curve_cubic(points):
def try_fit_curve_cubic(points):
    
    if len(points) < 4:
        raise "Too few points"
    
    # R = (ax³ + bx² + cx + d - y)²

    # 1. Multiply out the square - binominal formula!

    # [ S60  S50  S40  S30 ] [ a ]   [ S31 ]
    # [ S50  S40  S30  S20 ] [ b ] = [ S21 ]
    # [ S40  S30  S20  S10 ] [ c ]   [ S11 ]
    # [ S30  S20  S10  S00 ] [ d ]   [ S01 ]

    # [ +    -    +    -   ]
    # [ -    +    -    +   ]
    # [ +    -    +    -   ]
    # [ -    +    -    +   ]

    s00 = get_sum_x(points, 0)
    s10 = get_sum_x(points, 1)
    s20 = get_sum_x(points, 2)
    s30 = get_sum_x(points, 3)
    s40 = get_sum_x(points, 4)
    s50 = get_sum_x(points, 5)
    s60 = get_sum_x(points, 6)
    
    s31 = get_sum_x_3y(points)
    s21 = get_sum_x_2y(points)
    s11 = get_sum_x_y(points)
    s01 = get_sum_y(points)

    D = det_4x4([
        [ s60,  s50,  s40,  s30 ],
        [ s50,  s40,  s30,  s20 ],
        [ s40,  s30,  s20,  s10 ],
        [ s30,  s20,  s10,  s00 ]
    ])

    Da = det_4x4([
        [ s31,  s50,  s40,  s30 ],
        [ s21,  s40,  s30,  s20 ],
        [ s11,  s30,  s20,  s10 ],
        [ s01,  s20,  s10,  s00 ]
    ])
    
    Db = det_4x4([
        [ s60,  s31,  s40,  s30 ],
        [ s50,  s21,  s30,  s20 ],
        [ s40,  s11,  s20,  s10 ],
        [ s30,  s01,  s10,  s00 ]
    ])
    
    Dc = det_4x4([
        [ s60,  s50,  s31,  s30 ],
        [ s50,  s40,  s21,  s20 ],
        [ s40,  s30,  s11,  s10 ],
        [ s30,  s20,  s01,  s00 ]
    ])
    
    Dd = det_4x4([
        [ s60,  s50,  s40,  s31 ],
        [ s50,  s40,  s30,  s21 ],
        [ s40,  s30,  s20,  s11 ],
        [ s30,  s20,  s10,  s01 ]
    ])

    return [Da / D, Db / D, Dc / D, Dd / D]

def calc_error_cubic(points, curve):
    pass

def calc_error_quadratic(points, curve):
    pass

def calc_bezier_curve(points):
    bezier_points = [
        points[0],
        Point(-1, 1),
        Point(0, 3),
        points[len(points)- 1],
    ]

    return bezier_points

def main():

    points = [
        Point(-2.5,   2.64),
        Point(-2.0,   2.0),
        Point(-1.29,  1.8),
        Point(-0.52,  2.2),
        Point( 0.23,  2.53),
        Point( 0.67,  2.11),
        Point( 1.02,  1.34),
    ]

    bezier_points = calc_bezier_curve(points)

    # f(x) = 1x² + 0x + 0
    curve_cubic = try_fit_curve_cubic(points)
    
    print(curve_cubic)
    # f(x) = 1x³ + 0x² + 0x + 0
    # curve2 = [1, 0, 0, 0]

    open_window_and_show_results(points, curve_cubic, bezier_points)

main()