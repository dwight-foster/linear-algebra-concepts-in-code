import numpy as np

a = np.array([[2, 3], [4, 5]])
b = np.array([[2,3, 4], [5, 3, 7], [9, 2, 6]])

det_a = a[0][0] * a[1][1] - a[0][1] * a[1][0]
sign = 1
det_b = 0
for i in range(len(b[0])):
    num = b[0][i]
    cols = [0, 1, 2]
    cols.pop(i)
    det = b[1][cols[0]] * b[2][cols[1]] - b[1][cols[1]] * b[2][cols[0]]
    det_b += num * det * sign
    sign *= -1

print(det_a, np.linalg.det(a))
print(det_b, np.linalg.det(b))