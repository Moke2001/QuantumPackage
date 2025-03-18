import numpy as np
from qutip import *


def CliffordSet():
    f_0_dag=fcreate(2,0)
    f_0=f_0_dag.dag()
    f_1_dag=fcreate(2,1)
    f_1=f_1_dag.dag()

    braid=(1j*(np.pi/4)*(f_0_dag-f_0)*(f_1_dag+f_1)).expm()
    fswap=Qobj(np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,-1]]),dims=[[2,2],[2,2]])

    gamma_0=f_0_dag+f_0
    gamma_0_prem=1j*(f_0-f_0_dag)
    gamma_1=f_1_dag+f_1
    gamma_1_prem=1j*(f_1-f_1_dag)
    stabilizers=[tensor(identity(2),identity(2)),gamma_0,gamma_0_prem,gamma_1,gamma_1_prem,gamma_0*gamma_0_prem,gamma_1*gamma_1_prem,gamma_0*gamma_1,gamma_0_prem*gamma_1_prem,gamma_0*gamma_1_prem,gamma_0_prem*gamma_1,gamma_0*gamma_0_prem*gamma_1,gamma_0*gamma_0_prem*gamma_1_prem,gamma_0*gamma_1*gamma_1_prem,gamma_0_prem*gamma_1*gamma_1_prem,gamma_0*gamma_0_prem*gamma_1*gamma_1_prem]
    name=[[[0,0],[0,0]],
          [[1,0],[0,0]],
          [[0,1],[0,0]],
          [[0,0],[1,0]],
          [[0,0],[0,1]],
          [[1,1],[0,0]],
          [[0,0],[1,1]],
          [[1,0],[1,0]],
          [[0,1],[0,1]],
          [[1,0],[0,1]],
          [[0,1],[1,0]],
          [[1,1],[1,0]],
          [[1,1],[0,1]],
          [[1,0],[1,1]],
          [[0,1],[1,1]],
          [[1,1],[1,1]]]
    for temp in [gamma_0,gamma_0_prem,gamma_1,gamma_1_prem]:
        for i in range(len(stabilizers)):
            if (braid*temp*braid.dag()-stabilizers[i]).norm()<0.0001:

                print('+'+str(name[i]))
                break
            if (braid*temp*braid.dag()+stabilizers[i]).norm()<0.0001:
                print('-'+str(name[i]))
                break
