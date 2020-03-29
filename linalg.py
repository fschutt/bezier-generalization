class RegressionInput(object):
    def __init__(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c

class RegressionOutput(object):
    def __init__(self, x1, x2):
        self.x1 = x1
        self.x2 = x2

# @returns RegressionOutput
def solve_equations(equations):

    num_equations = len(equations)
    
    c_sum = 0.0
    a_sum = 0.0
    b_sum = 0.0
    ac_sum = 0.0
    bc_sum = 0.0
    ab_sum = 0.0

    for eq in equations:
        c_sum += eq.c
        a_sum += eq.a
        b_sum += eq.b
        ac_sum += eq.a * eq.c
        bc_sum += eq.b * eq.c
        ab_sum += eq.a * eq.b

    c_mean = c_sum / num_equations
    a_mean = a_sum / num_equations
    b_mean = b_sum / num_equations

    # Calculate sum of squares of deviations of data points from their sample mean (DEVSQ)
    devsq_c = 0.0
    devsq_a = 0.0
    devsq_b = 0.0

    for eq in equations:
        devsq_c += pow(eq.c - c_mean, 2)
        devsq_a += pow(eq.a - a_mean, 2)
        devsq_b += pow(eq.b - b_mean, 2)

    # Calculate the x1 and x2 values

    ca = ac_sum - a_sum * c_sum / num_equations
    cb = bc_sum - b_sum * c_sum / num_equations
    ab = ab_sum - a_sum * b_sum / num_equations

    x1 = (devsq_b*ca-ab*cb) / (devsq_a*devsq_b-ab*ab)
    x2 = (devsq_a*cb-ab*ca) / (devsq_a*devsq_b-ab*ab)

    return RegressionOutput(x1, x2)

def calc_error_squared(equations, output):
    error = 0.0
    for eq in equations:
        error += pow(eq.c - (eq.a * output.x1 + eq.b * output.x2), 2)
    return error

def calc_error_normal(equations, output):
    error = 0.0
    for eq in equations:
        error += eq.c - (eq.a * output.x1 + eq.b * output.x2)
    return error

def main():

    equations = [
        #               a*x1                +  b*x2                 = y
        RegressionInput(0,                     0,                     0),
        RegressionInput(0.355116901775958,     0.074355530922201,     3.70277585006972),
        RegressionInput(0.444418152930507,     0.219266287478505,     6.0607006535147),
        RegressionInput(0.363145347644357,     0.386147834850555,     7.11791957320285),
        RegressionInput(0.198903631449205,     0.442769595436206,     6.38607945041053),
        RegressionInput(0.079946019237193,     0.363442774579523,     4.55677830027078),
        RegressionInput(0,                     0,                     0),
    ]

    output = solve_equations(equations)

    print("output:")
    print("\tx1: " + str(output.x1))
    print("\tx2: " + str(output.x2))
    print("error squared: " + str(calc_error_squared(equations, output)))
    print("error normal: " + str(calc_error_normal(equations, output)))

    # Now try it with x1 = 9, x2 = 10 to compare
    print("\n-----\n")
    fake_output = RegressionOutput(9, 0, 10, 0)
    print("output:")
    print("\tx1: " + str(fake_output.x1))
    print("\tx2: " + str(fake_output.x2))
    print("error squared: " + str(calc_error_squared(equations, fake_output)))
    print("error normal: " + str(calc_error_normal(equations, fake_output)))

main()