import numpy as np


#%%  KEY：计算有限域上向量在基矢组上的表示
def finite_field_solve(vectors, target):
    vectors = np.array(vectors,dtype=int)
    target = np.array(target,dtype=int)
    m = len(vectors)
    n = len(target)

    # 构造增广矩阵 (n行, m+1列)
    augmented = []
    for i in range(n):
        row = [vec[i] for vec in vectors] + [target[i]]
        augmented.append(row.copy())

    # 高斯消元主循环
    rank = 0
    for col in range(m):  # 遍历每一列
        # 寻找主元行
        pivot = -1
        for r in range(rank, n):
            if augmented[r][col] == 1:
                pivot = r
                break

        if pivot == -1:  # 该列无主元，跳过
            continue

        # 交换行到当前秩位置
        augmented[rank], augmented[pivot] = augmented[pivot], augmented[rank]

        # 消去其他行的当前列元素
        for r in range(n):
            if r != rank and augmented[r][col] == 1:
                augmented[r] = [(a + b) % 2 for a, b in zip(augmented[r], augmented[rank])]

        rank += 1

    # 反向代入求解
    solution = [0] * m
    for r in range(rank):
        # 定位主元列
        pc = -1
        for c in range(m):
            if augmented[r][c] == 1:
                pc = c
                break
        if pc == -1:
            continue

        # 计算该变量的值
        val = augmented[r][-1]
        for c in range(pc + 1, m):
            val ^= (augmented[r][c] & solution[c])

        solution[pc] = val

    # 验证解的正确性
    for i in range(n):
        sum_bit = 0
        for j in range(m):
            sum_bit ^= (solution[j] & vectors[j][i])
        if sum_bit != target[i]:
            return None

    return solution