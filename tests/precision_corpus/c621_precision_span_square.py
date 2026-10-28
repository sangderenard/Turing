def f(x: Precision[2], y: Precision[2], n):
    for i in range(n):
        z = x[i]
        y[i] = z * z
    return y
