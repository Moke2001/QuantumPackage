import copy
from multiprocessing import Pool
import numpy as np
from MokeQuantumComputation.CliffordSimulator.FermionicComputer import FermionicComputer
from MokeQuantumComputation.Code.ClassicalCode.EuclideanCode import EuclideanCode
from MokeQuantumComputation.Code import FermionicCode


# %%  USER：计算量子线路执行结果
def simulation(p, sample_number,code,stabilizers):
    # %%  SECTION：创建不同的纠错码，初始化态矢
    """""
    态的初始化和纠错码构造
    考虑布局为：内存a，内存b，处理器a，处理器b，辅助
    """""

    # %%  SECTION：进行逻辑线路
    n_logical = len(code.logical_operator_list_x)
    logical_p = 0
    computer = FermionicComputer()
    computer.define(code.number_qubit, p)
    computer.initialize(copy.deepcopy(stabilizers))

    for j in range(sample_number):
        ##  初始化
        computer.noise_recoder_x = []
        computer.noise_recoder_z = []
        computer.clear_x = []
        computer.clear_z = []
        for i in range(len(computer.stabilizers)):
            computer.stabilizers[i].coff = 1

        for i in range(10):
            computer.loss(range(code.number_qubit))
            computer.correct(code, 'common', range(code.number_checker))

        for i in range(1, n_logical + 1):
            if computer.stabilizers[-i].coff == -1:
                logical_p += 1
                break

    return logical_p


if __name__ == '__main__':
    ##  创建内存纠错码
    code = FermionicCode()
    code.linear_combine(EuclideanCode(2, 3, 2))
    code.get_logical_operators()
    stabilizers = []
    number_qubit = code.number_qubit
    ##  初始化
    for i in range(len(code.check_list)):
        stabilizers.append(code.check_list[i].copy())

    for i in range(len(code.logical_operator_list_x)):
        stabilizers.append(code.logical_operator_list_x[i].copy())

    physical_p_list =  np.linspace(0.05, 0.15, 10)
    sample_number = 10
    cycle_number = 20
    for p in physical_p_list:
        logical_error_rate_list = []
        for cycle in range(cycle_number):
            print('cycle',cycle)
            with Pool(processes=20) as pool:
                results = [pool.apply_async(simulation, args=(p_temp, sample_number,code,stabilizers)) for p_temp in [p]*20]
                final_results = [result.get() for result in results]
            logical_error_rate_list.append(np.sum(final_results)/(len(final_results)*sample_number))
        print(p,':',np.mean(logical_error_rate_list),np.std(logical_error_rate_list))

    print(final_results)