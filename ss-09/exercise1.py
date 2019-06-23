def fx(x, y):
    return (2*(x - y))/(x** + 2*y** - 2* x * y - 2*y + 2)

def fy(x, y):
    return (4*y - 2*x - 2)/(x** + 2*y** - 2* x * y - 2*y + 2)

if __name__ == '__main__':
    alpha = 0.5

    x0 = -5
    y0 = -5

    dx1 = fx(x0, y0)
    dy1 = fy(x0, y0)
    x1 = x0 - alpha * dx1
    y1 = y0 - alpha * dy1

    dx2 = fx(x1, y1)
    dy2 = fy(x1, y1)
    x2 = x1 - alpha * dx2
    y2 = y1 - alpha * dy2

    dx3 = fx(x2, y2)
    dy3 = fy(x2, y2)
    x3 = x2 - alpha * dx3
    y3 = y2 - alpha * dy3

    print("Gradient Descent")
    print("after iteration 1")
    print(f"x1: {x1} x2: {y1}")
    print("after iteration 2")
    print(f"x1: {x2} x2: {y2}")
    print("after iteration 3")
    print(f"x1: {x3} x2: {y3}")

    beta = 0.9
    vx0 = 0
    vy0 = 0

    dmx1 = fx(x0, y0)
    dmy1 = fy(x0, y0)
    vx1 = beta * vx0 - alpha * dmx1
    vy1 = beta * vy0 - alpha * dmy1
    mx1 = x0 + vx1
    my1 = y0 + vy1

    dmx2 = fx(vx1, vy1)
    dmy2 = fy(vx1, vy1)
    vx2 = beta * vx1 - alpha * dmx2
    vy2 = beta * vy1 - alpha * dmy2
    mx2 = mx1 + vx2
    my2 = my1 + vy2

    dmx3 = fx(vx2, vy2)
    dmy3 = fy(vx2, vy2)
    vx3 = beta * vx2 - alpha * dmx3
    vy3 = beta * vy2 - alpha * dmy3
    mx3 = mx2 + vx3
    my3 = my2 + vy3

    print("With momentum")
    print("after iteration 1")
    print(f"x1: {mx1} x2: {my1}")
    print("after iteration 2")
    print(f"x1: {mx2} x2: {my2}")
    print("after iteration 3")
    print(f"x1: {mx3} x2: {my3}")

