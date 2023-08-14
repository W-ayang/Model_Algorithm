
import numpy as np
import time

# 生成两个大矩阵
n = 200
A = np.random.rand(n, n)
B = np.random.rand(n, n)

# 使用 @ 进行矩阵乘法
start_time = time.time()
C = A @ B
end_time = time.time()
print("使用 @ 进行矩阵乘法的时间：", end_time - start_time)

# 使用循环进行矩阵乘法
start_time = time.time()
C = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        for k in range(n):
            C[i][j] += A[i][k] * B[k][j]
end_time = time.time()
print("使用循环进行矩阵乘法的时间：", end_time - start_time)

