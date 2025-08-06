import itertools
from multiprocessing import Process
import galois
import numpy as np
from MokeQuantumComputation.Helper.FiniteFieldSolve import FiniteFieldSolve

upper = None
lower = None
GF=galois.GF(2**1)


#%%  KEY：计算quantum stabilizer code的码距
"""
input.matrix：np.array of GF(2)对象，校验矩阵
input.gauge_list：list of np.array of GF(2)对象，规范子
output：int对象，码距
"""
def QuantumCodeDistance(matrix,gauge_list):
    global upper
    global lower
    GF = galois.GF(2 ** 1)
    matrix=GF(np.array(matrix,dtype=int))
    gauge_list_temp=gauge_list
    gauge_list=[]
    for vec in gauge_list_temp:
        gauge_list.append(GF(np.array(vec,dtype=int)))

    center_list=matrix.null_space()
    for vec in gauge_list:
        matrix=np.vstack([matrix,vec])

    # 提交任务给进程池
    p0 = Process(target=upper_bound, args=(matrix,center_list))
    p1 = Process(target=lower_bound, args=(matrix, center_list))
    p0.start()
    p1.start()
    p0.join()
    p1.join()

    return lower


def upper_bound(matrix,center_list):
    global upper
    for num in range(1,matrix.shape[1]):
        for each in itertools.combinations(center_list,num):
            temp=None
            for vec in each:
                if temp is None:
                    temp=vec
                else:
                    temp=temp+vec
            if FiniteFieldSolve(matrix, temp) is None:
                if upper is None or upper > np.count_nonzero(temp):
                    upper = np.count_nonzero(temp)
                    print('upperbound:', upper)
            if lower == upper and upper is not None:
                break
        if lower == upper and lower is not None:
            break

def lower_bound(matrix,center_list):
    global lower
    global upper
    pos=range(matrix.shape[1])
    for num in range(1,matrix.shape[1]):
        lower=num
        print('lowerbound:', lower)
        for each in itertools.combinations(pos,num):
            temp=GF(np.zeros(matrix.shape[1],dtype=int))
            temp[list(each)]=1
            if np.count_nonzero(matrix@temp)==0:
                if FiniteFieldSolve(matrix, temp) is None:
                    upper = np.count_nonzero(temp)
                    print('upperbound:', upper)
            if lower==upper:
                break
        if lower==upper:
            break

