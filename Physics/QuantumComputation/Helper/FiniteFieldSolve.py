import numpy as np


#%%  KEY：计算有限域上向量在基矢组上的表示
"""
input.vectors：list of np.array of GF(2)对象，向量组
input.target：np.array of GF(2)对象，目标向量
ouput：None or list对象，vectors上的系数解
"""
def FiniteFieldSolve(vectors, target):
    #%%  SECTION：格式化

    vectors = np.array(vectors, dtype=int)
    target = np.array(target, dtype=int)
    m, n = vectors.shape[0], target.shape[0]


    #%%  SECTION：构造增广矩阵

    A = np.vstack(vectors).T
    aug = np.hstack([A, target.reshape(-1, 1)])  # n x (m+1)
    rank = 0

    for col in range(m):
        # 找主元
        rows_with_one = np.where(aug[rank:, col] == 1)[0]
        if rows_with_one.size == 0:
            continue
        pivot = rank + rows_with_one[0]
        # 行交换
        if pivot != rank:
            aug[[rank, pivot]] = aug[[pivot, rank]]
        # 消元
        mask = (aug[:, col] == 1) & (np.arange(aug.shape[0]) != rank)
        aug[mask] ^= aug[rank]
        rank += 1


    #%%  SECTION：反向代入

    solution = np.zeros(m, dtype=int)
    for r in range(rank):
        row = aug[r, :m]
        idx = np.where(row == 1)[0]
        if idx.size == 0:
            continue
        pc = idx[0]
        val = aug[r, -1]
        if pc + 1 < m:
            val ^= np.dot(row[pc+1:], solution[pc+1:]) % 2
        solution[pc] = val


    #%%  SECTION：验证并输出结果

    if not np.all((np.dot(A, solution) % 2) == target):
        return None
    return solution.tolist()