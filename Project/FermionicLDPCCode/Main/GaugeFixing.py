import copy
import galois
import numpy as np
from Physics.QuantumComputation.Helper.FiniteFieldSolve import FiniteFieldSolve


GF=galois.GF(2**1)


def set_minus(basis2, basis1):
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


def GaugeFixing():
    temp=np.zeros(48,dtype=int)
    check_0 = [0,1,2,3]
    check_1 = [2,3,4,5,7,8]
    check_2 = [1,3,5,6]
    check_3 = [4,7,10,11]
    check_4 = [5,6,8,9,12,13]
    check_origin_0 = [10,11,14,15]
    check_origin_1 = [7,8,11,12,15,16]
    check_origin_2 = [12,13,16,17]
    check_origin_3 = [9,13,17,18]
    check__origin_4 = [19,20,24,25]
    check__origin_5 = [19,22,24,27]
    check__origin_6 = [21,23,26,28]
    check_gauge_0 = [29,30,37,38]
    check_gauge_1 = [31,32,39,40]
    check_gauge_2 = [33,34,45,46]
    check_gauge_3 = [41,42,43,44]
    check_gauge_4 = [35,36,47,48]

    check_modified_0=[10,11,14,15,29,30]
    check_modified_1=[7,8,11,12,15,16,31,32]
    check_modified_2=[12,13,16,17,10,11,14,15,33,34,41,42]
    check_modified_3=[9,12,13,16,18,35,36]
    check_modified_4=[19,20,24,25,37,38]
    check_modified_5=[19,22,24,27,45,46]
    check_modified_6=[21,23,26,28,47,48]

    check_measure_0=[14,19,29,37,33,45]
    check_measure_1=[15,20,30,31,38,39,41,43]
    check_measure_2=[16,21,32,40,35,47,42,44]
    check_measure_3=[17,22,34,46]
    check_measure_4=[18,23,36,48]
    check_origin_list=[check_0,check_1,check_2,check_3,check_4,
                       check_origin_0,check_origin_1,check_origin_2,check_origin_3,check__origin_4,check__origin_5,check__origin_6,
                       ]
    check_merge_list=[check_0,check_1,check_2,check_3,check_4,
                      check_modified_0,check_modified_1,check_modified_2,check_modified_3,check_modified_4,check_modified_5,check_modified_6,
                      check_measure_0,check_measure_1,check_measure_2,check_measure_3,check_measure_4,
                      check_gauge_0,check_gauge_1,check_gauge_2,check_gauge_3,check_gauge_4
                      ]
    origin_generators=[]
    for i in range(len(check_origin_list)):
        temp=GF(np.zeros(49,dtype=int))
        temp[check_origin_list[i]]=1
        origin_generators.append(temp)
    origin_generators=GF(np.array(origin_generators,dtype=int))
    merge_generators=[]
    for i in range(len(check_merge_list)):
        temp=GF(np.zeros(49,dtype=int))
        temp[check_merge_list[i]]=1
        merge_generators.append(temp)

    merge_generators=GF(np.array(merge_generators,dtype=int))

    gauge_generators=copy.deepcopy(merge_generators)
    for i in range(5,12):
        temp=GF(np.zeros(49,dtype=int))
        temp[check_origin_list[i]]=1
        gauge_generators=np.vstack([gauge_generators,temp])
    prep_generators=set_cap(origin_generators,set_minus(merge_generators,merge_generators.null_space()))
    meas_generators = set_cap(merge_generators, set_minus(origin_generators, origin_generators.null_space()))

    for i in range(29,49):
        for j in range(i,49):
            temp = GF(np.zeros(49, dtype=int))
            temp[[i,j]] = 1
            gauge_generators = GF(np.vstack([gauge_generators, temp]))
    temp = GF(np.zeros(49, dtype=int))
    temp[23] = 1
    gauge_generators = GF(np.vstack([gauge_generators, temp]))
    temp = GF(np.zeros(49, dtype=int))
    temp[[14,15,16,17,18]] = 1
    gauge_generators=GF(np.vstack([gauge_generators, temp]))
    error=GF(np.zeros(49,dtype=int))
    error[[10,11,31,39]]=1
    logic=GF(np.zeros(49,dtype=int))
    logic[[21,23,47,48]]=1

    stabilizer_generators=set_cap(gauge_generators,gauge_generators.null_space())
    L_gauge_generators=set_minus(stabilizer_generators,gauge_generators)
    L_bare_generators=set_minus(gauge_generators,gauge_generators.null_space())
    L_dress_generators=np.vstack([L_bare_generators,gauge_generators])
    L_merge_generators=set_minus(merge_generators,merge_generators.null_space())

    print(FiniteFieldSolve(gauge_generators,error))
    print(FiniteFieldSolve(stabilizer_generators,error))
    print(FiniteFieldSolve(L_dress_generators,error))
    print(FiniteFieldSolve(L_bare_generators,error))
    print(FiniteFieldSolve(L_gauge_generators,error))
    pass


if __name__ == '__main__':
    GaugeFixing()