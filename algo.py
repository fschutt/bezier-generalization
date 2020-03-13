
        

def get_sum_x(point_array, power):
    sum = 0
    
    for p in point_array:
        sum += pow(p.x, power)

    return sum


points = [
    Point(2, 0)
    Point(3, 4)
]

print(get_sum_x(points, 2))