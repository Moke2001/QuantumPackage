import numpy as np
import galois

from Physics.QuantumComputation.Helper.FiniteFieldSolve import FiniteFieldSolve


def LinearSpaceDiff(basis1, basis2):
    GF2 = galois.GF(2)

    # 转化为GF2数组
    basis1 = GF2(basis1)
    basis2 = GF2(basis2)

    intersect=set_cap(basis1, basis2)
    if len(intersect)==0:
        return basis1

    result=[]
    for i in range(len(basis1)):
        rank=np.linalg.matrix_rank(intersect)
        intersect=np.vstack((intersect,basis1[i]))
        if np.linalg.matrix_rank(intersect)>rank:
            result.append(basis1[i])

    return GF2(np.array(result,dtype=int))

def set_cap(basis1, basis2):
    GF2 = galois.GF(2)

    # 转化为GF2数组
    basis1 = GF2(basis1)
    basis2 = GF2(basis2)

    m = basis1.shape[0]
    k = basis2.shape[0]

    # 解方程 basis1^T * x + basis2^T * y = 0 (表示交集向量)
    # 构造矩阵 [basis1^T | basis2^T]
    aug_matrix = GF2(np.concatenate((basis1.T, basis2.T), axis=1))

    # 计算零空间（解空间）
    nullspace = aug_matrix.null_space()

    # 从零空间中取对应于每一组自变量两部分中对应于 basis1 的分量在这里获得解的a区段
    ab_space = nullspace[:, :m]

    # 将系数乘以 basis1 得到具体解（交集）
    if len(ab_space) == 0:
        return GF2.Zeros((0, basis1.shape[1]))

    intersection_vectors = ab_space @ basis1

    # 产生基向量的线性无关组合，即每种向量的不同的高斯行组合样本
    # 这里要消除重复或多余的向量求其产生的张成空间作为交集

    # 计算行相关即进行矩阵的高斯消元
    rref_intersection = intersection_vectors.row_reduce()

    # 除去全0行的行（即为没有有用基底捕获的时候）
    nz_mask = np.any(rref_intersection != 0, axis=1)
    rref_basis = rref_intersection[nz_mask]

    return rref_basis

if __name__ == '__main__':
    # 测试用例
    GF2 = galois.GF(2)

    # 用例1: 基本测试
    basis1 = GF2([[1, 0, 0], [0, 1, 0]])
    basis2 = GF2([[1, 1, 0]])
    diff = LinearSpaceDiff(basis1, basis2)
    print("用例1结果:\n", diff)

    # 用例2: 交集等于第一个空间
    basis1 = GF2([[1, 0, 0], [0, 1, 0]])
    basis2 = GF2([[1, 0, 0], [0, 1, 0], [1, 1, 0]])
    diff = LinearSpaceDiff(basis1, basis2)
    print("用例2结果:\n", diff)

    # 用例3: 无交集
    basis1 = GF2([[1, 0, 0]])
    basis2 = GF2([[0, 1, 0]])
    diff = LinearSpaceDiff(basis1, basis2)
    print("用例3结果:\n", diff)

    # 用例4: 子空间
    basis1 = GF2([[1, 0, 0], [0, 1, 0], [0, 1, 1]])
    basis2 = GF2([[1, 1, 0]])
    diff = LinearSpaceDiff(basis1, basis2)
    print("用例4结果:\n", diff)

