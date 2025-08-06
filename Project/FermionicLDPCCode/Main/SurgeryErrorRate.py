import copy
from multiprocessing import Pool
import numpy as np
from MokeQuantumComputation.CliffordSimulator.FermionicComputer import FermionicComputer
from MokeQuantumComputation.Code import ProjectiveCode
from MokeQuantumComputation.Code import FermionicCode
from MokeQuantumComputation.Code.QuantumCode.Fermion.FermionicColorCode import FermionicColorCode
from MokeQuantumComputation.Code.QuantumCode.Fermion.MajoranaOperator import MajoranaOperator
from MokeQuantumComputation.Code.QuantumCode.Fermion.ZechuanLatticeSurgery import ZechuanLatticeSurgery


#%%  USER：计算量子线路执行结果
def simulation(p,sample_number,code,stabilizers_0,stabilizers_1):

    #%%  SECTION：创建不同的纠错码，初始化态矢
    """""
    态的初始化和纠错码构造
    考虑布局为：内存a，内存b，处理器a，处理器b，辅助
    """""

    #%%  SECTION：进行逻辑线路
    logical_p=0
    computer = FermionicComputer()
    computer.define(code.number_qubit, p)
    if np.random.rand()<0.5:
        computer.initialize(copy.deepcopy(stabilizers_0))
    else:
        computer.initialize(copy.deepcopy(stabilizers_1))
    n_logical = 7
    for j in range(sample_number):
        ##  初始化
        computer.noise_recoder_x=[]
        computer.noise_recoder_z=[]
        computer.clear_x=[]
        computer.clear_z=[]
        for i in range(len(computer.stabilizers)):
            computer.stabilizers[i].coff=1

        for i in range(10):
            computer.depolarize(range(code.number_qubit))
            computer.correct(code,'common',range(code.number_checker))

        for i in range(1,n_logical+1):
            if computer.stabilizers[-i].coff==-1:
                logical_p+=1
                break

    return logical_p


if __name__ == '__main__':
    ##  创建内存纠错码
    ##  创建内存纠错码
    code_memory = FermionicCode()
    code_memory.linear_combine(ProjectiveCode(3))

    opx0 = MajoranaOperator([0, 7, 25, 41, 51], [], 1)
    opx1 = MajoranaOperator([1, 11, 47, 50, 62], [], 1)
    opx2 = MajoranaOperator([2, 10, 27, 43, 54], [], 1)
    opx3 = MajoranaOperator([3, 13, 22, 31, 58], [], 1)
    opz0 = MajoranaOperator([], [0, 7, 25, 41, 51], 1)
    opz1 = MajoranaOperator([], [1, 11, 47, 50, 62], 1)
    opz2 = MajoranaOperator([], [2, 10, 27, 43, 54], 1)
    opz3 = MajoranaOperator([], [3, 13, 22, 31, 58], 1)
    code_memory.logical_operator_list_x = [opx0, opx1, opx2, opx3]
    code_memory.logical_operator_list_z = [opz0, opz1, opz2, opz3]
    n_memory = code_memory.number_qubit

    ##  创建处理器纠错码
    code_processor = FermionicColorCode(5)
    n_processor = code_processor.number_qubit

    ##  融合码
    code = ZechuanLatticeSurgery(code_memory,0)
    n_ancilla = code.number_qubit - n_processor - n_memory  # 辅助粒子数
    n_total = n_processor * 2 + n_ancilla + n_memory * 2  # 所有粒子数目

    stabilizers_plus = []

    ##  初始化
    for i in range(len(code.check_list)):
        stabilizers_plus.append(code.check_list[i].copy())

    temp_x = code_memory.logical_operator_list_x[0].x_vector
    op0 = MajoranaOperator(temp_x, [], 1)

    temp_x_1 = code_memory.logical_operator_list_x[1].x_vector
    op1 = MajoranaOperator(temp_x_1, [], 1)
    temp_x_2 = temp_x_1 + n_memory
    op2 = MajoranaOperator(temp_x_2, [], 1)

    temp_x_3 = code_memory.logical_operator_list_x[2].x_vector
    op3 = MajoranaOperator(temp_x_3, [], 1)
    temp_x_4 = temp_x_3 + n_memory
    op4 = MajoranaOperator(temp_x_4, [], 1)

    temp_x_5 = code_memory.logical_operator_list_x[3].x_vector
    op5 = MajoranaOperator(temp_x_5, [], 1)
    temp_x_6 = temp_x_5 + n_memory
    op6 = MajoranaOperator(temp_x_6, [], 1)

    stabilizers_plus = stabilizers_plus + [op0, op1, op2, op3, op4, op5, op6]

    number_qubit = code.number_qubit
    ##  初始化
    for i in range(len(code.check_list)):
        stabilizers_plus.append(code.check_list[i].copy())
    stabilizers_zero=stabilizers_plus.copy()
    for i in range(len(code.logical_operator_list_x)):
        stabilizers_plus.append(code.logical_operator_list_x[i].copy())
        temp=MajoranaOperator(code.logical_operator_list_x[i].x_vector,code.logical_operator_list_x[i].x_vector , 1)
        stabilizers_zero.append(temp)

    physical_p_list = np.linspace(0.01, 0.05, 10)
    sample_number = 100
    cycle_number = 20
    for p in physical_p_list:
        logical_error_rate_list = []
        for cycle in range(cycle_number):
            print('cycle',cycle)
            with Pool(processes=20) as pool:
                results = [pool.apply_async(simulation, args=(p_temp, sample_number, code, stabilizers_plus,stabilizers_zero)) for p_temp in [p] * 20]
                final_results = [result.get() for result in results]
            logical_error_rate_list.append(np.sum(final_results)/(len(final_results)*sample_number))
        print(p,':',np.mean(logical_error_rate_list),np.std(logical_error_rate_list))


    print(final_results)
