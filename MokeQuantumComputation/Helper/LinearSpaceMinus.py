import galois
import numpy as np

from MokeQuantumComputation.Helper.LinearSpaceCap import LinearSpaceCap


def LinearSpaceMinus(basis1, basis2):
    GF2 = galois.GF(2)

    # 转化为GF2数组
    basis1 = GF2(basis1)
    basis2 = GF2(basis2)

    intersect=LinearSpaceCap(basis1, basis2)
    if len(intersect)==0:
        return basis1

    result=[]
    for i in range(len(basis1)):
        rank=np.linalg.matrix_rank(intersect)
        intersect=np.vstack((intersect,basis1[i]))
        if np.linalg.matrix_rank(intersect)>rank:
            result.append(basis1[i])

    return GF2(np.array(result,dtype=int))