## 测量函数
import numpy as np
from matlab import engine
from qutip import *
import os


def measure(psi,operator):
    eigenstate_list = operator.eigenstates()
    eigenvalue_list=operator.eigenvalues()
    P_list=[]
    for i in range(len(eigenstate_list)):
        P_list.append(eigenstate_list[i]*psi*psi.dag()*eigenvalue_list[i].dag())
    array_temp = range(len(eigenstate_list))
    order = np.random.choice(a=array_temp, p=P_list)
    return eigenstate_list[order]*eigenstate_list[order].dag()*psi,eigenvalue_list[order]