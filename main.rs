#[derive(Debug, Copy, Clone)]
pub enum Error {
    TooFewPoints,
}

// 2D point
#[derive(Debug, Copy, Clone)]
pub struct Point { x: f64, y: f64 }

// Inputs for one formula in the form of y = a*x1 + b*x2
#[derive(Copy, Clone)]
struct RegressionInput { y: f64, a: f64, b: f64 }

const REGRESSION_INPUT_ZEROED: RegressionInput = RegressionInput { y: 0.0, a: 0.0, b: 0.0 };

// Output of the 2-variable regression algorithm
struct RegressionOutput {
    x1: f64,
    error_x1: f64,
    x2: f64,
    error_x2: f64,
}

pub fn estimate_bezier_curve(points: &[Point]) -> Result<[Point;4], Error> {

    if points.len() < 4 {
        return Err(Error::TooFewPoints);
    }

    let p0 = unsafe { *points.get_unchecked(0) };
    let p3 = unsafe { *points.get_unchecked(points.len() - 1) };

    #[inline]
    fn distance(a: &Point, b: &Point) -> f64 { 
        let delta_x = b.x - a.x;
        let delta_y = b.y - a.y;
        (delta_x * delta_x + delta_y * delta_y).sqrt() 
    }

    // Calculate the length of each line and divide by the sum to get the 
    // chord-length estimated "t" value for each point
    let mut t_values = vec![0.0;points.len()];
    let mut total_len = 0.0;
    for (i, (a, b)) in points.iter().zip(points.iter().skip(1)).enumerate() {
        let dst = distance(a, b);
        unsafe { *t_values.get_unchecked_mut(i + 1) = dst; }
        total_len += dst;
    }

    let mut cur_len = 0.0;
    for value in &mut t_values {
        let t = (cur_len + *value) / total_len;
        cur_len += *value;
        *value = t;
    }

    // Since the cubic bezier curve has the form of:
    // pi.x = (1-t)³p0.x + 3(1-t)²tp1.x + 3(1-t)t²p2.x + t³p3.x
    //
    // ... and we know t, p0, p3 and x - we can solve:
    // pi.x - (1-t)³p0.x - t³p3.x = 3(1-t)²t*p1.x + 3(1-t)t²*p2.x
    //
    // Since we know t, we can calculate the coefficients and simplify down to:
    // pi.x - (1-t)³p0.x - t³p3.x = a*p1.x + b*p2.x
    //
    // and calculate the term on the other side to reduce it to "x1":
    // x1 = a*p1.x + b*p2.x
    // 
    let mut regression_x_inputs = vec![REGRESSION_INPUT_ZEROED; t_values.len()];
    let mut regression_y_inputs = vec![REGRESSION_INPUT_ZEROED; t_values.len()];

    for (i, (t, pi)) in t_values.iter().zip(points.iter()).enumerate() {

        let one_minus = 1.0 - t;
        let one_minus_t3 = one_minus * one_minus * one_minus;
        let t3 = t*t*t;
        let x1 = pi.x - (one_minus_t3 * p0.x) - t3*p3.x;
        let y1 = pi.y - (one_minus_t3 * p0.y) - t3*p3.y;
        let a = 3.0 * one_minus* one_minus * t;
        let b = 3.0 * one_minus * t*t;

        unsafe {        
            *regression_x_inputs.get_unchecked_mut(i) = RegressionInput { y: x1, a, b };
            *regression_y_inputs.get_unchecked_mut(i) = RegressionInput { y: y1, a, b };
        }
    }

    let solved_x = quadratic_regression(&regression_x_inputs);
    let solved_y = quadratic_regression(&regression_y_inputs);

    Ok([
        p0,
        Point { x: solved_x.x1, y: solved_y.x1 },
        Point { x: solved_x.x2, y: solved_y.x2 },
        p3,
    ])
}

fn quadratic_regression(equations: &[RegressionInput]) -> RegressionOutput {
    
    let num_equations = equations.len() as f64;

    let mut y_sum = 0.0;
    let mut a_sum = 0.0;
    let mut b_sum = 0.0;
    let mut ay_sum = 0.0; // sum of a * y
    let mut by_sum = 0.0; // sum of a * y
    let mut ab_sum = 0.0; // sum of a * y

    for eq in equations { 
        y_sum += eq.y; 
        a_sum += eq.a; 
        b_sum += eq.b; 
        ay_sum += eq.a * eq.y;
        by_sum += eq.b * eq.y;
        ab_sum += eq.a * eq.b;
    }

    let y_mean = y_sum / num_equations;
    let a_mean = a_sum / num_equations;
    let b_mean = b_sum / num_equations;

    // Calculate sum of squares of deviations of data points from their sample mean (DEVSQ)
    let mut devsq_y = 0.0;
    let mut devsq_a = 0.0;
    let mut devsq_b = 0.0;
    let mut devsq_y_minus_a = 0.0;
    let mut devsq_y_minus_b = 0.0;
    let mut devsq_a_minus_b = 0.0;

    for eq in equations {
        let y_diff = eq.y - y_mean;
        let a_diff = eq.a - a_mean;
        let b_diff = eq.b - b_mean;
        devsq_y += y_diff * y_diff;
        devsq_a += a_diff * a_diff;
        devsq_b += b_diff * b_diff;
        devsq_y_minus_a += y_diff * a_diff;
        devsq_y_minus_b += y_diff * b_diff;
        devsq_a_minus_b += b_diff * a_diff;
    }

    // Calculate the a and b values plus their error rya, ryb and rab

    let ya = ay_sum - a_sum * y_sum / num_equations;
    let yb = by_sum - b_sum * y_sum / num_equations;

    let aa = devsq_a;
    let ab = ab_sum - a_sum * b_sum / num_equations;
    let bb = devsq_b;
    
    let r_ya = devsq_y_minus_a / (devsq_y * devsq_a).sqrt(); // sample correlation coefficient of y -> a
    let r_yb = devsq_y_minus_b / (devsq_y * devsq_b).sqrt(); // sample correlation coefficient of y -> a
    let r_ab = devsq_a_minus_b / (devsq_a * devsq_b).sqrt(); // sample correlation coefficient of b -> a

    let div = aa*bb-ab*ab;
    let r_div = 1.0 - r_ab*r_ab;

    RegressionOutput {
        x1: (bb*ya-ab*yb) / div,
        error_x1: (r_ya-r_yb*r_ab) / r_div,
        x2: (aa*yb-ab*ya) / div,
        error_x2: (r_yb - r_ya * r_ab) / r_div,
    }
}

fn main() -> Result<(), Error> {

    let points = [
        Point { x: 7.5,   y: 2.64 },
        Point { x: 8.0,   y: 2.0  },
        Point { x: 8.71,  y: 1.8  },
        Point { x: 9.48,  y: 2.2  },
        Point { x: 10.23, y: 2.53 },
        Point { x: 10.67, y: 2.11 },
        Point { x: 11.02, y: 1.34 },
    ];

    // expected:
    //
    // Point { x: 7.5,    y: 2.64  }
    // Point { x: 9.0,    y: 0.0   }
    // Point { x: 10.0,   y: 4.5   }
    // Point { x: 11.02,  y: 1.34  }

    println!("result: {:#?}", estimate_bezier_curve(&points)?);
    Ok(())
}