from multiprocessing import Pool
import numpy as np
from MokeQuantumComputation.CliffordSimulator.FermionicComputer import FermionicComputer
from MokeQuantumComputation.Code import ProjectiveCode
from MokeQuantumComputation.Code import FermionicCode
from MokeQuantumComputation.Code.QuantumCode.Fermion.FermionicColorCode import FermionicColorCode
from MokeQuantumComputation.Code.QuantumCode.Fermion.MajoranaOperator import MajoranaOperator
from MokeQuantumComputation.Code.QuantumCode.Fermion.ZechuanLatticeSurgery import ZechuanLatticeSurgery

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
n_memory = code_memory.number_qubit  # 内存的粒子数

##  创建处理器纠错码
code_processor = FermionicColorCode(5)
n_processor = code_processor.number_qubit  # 处理器的粒子数

##  融合码
code_merge_0 = ZechuanLatticeSurgery(code_memory,0)
code_merge_1 = ZechuanLatticeSurgery(code_memory,1)
code_merge_2 = ZechuanLatticeSurgery(code_memory,2)
code_merge_3 = ZechuanLatticeSurgery(code_memory,3)
n_ancilla=code_merge_0.number_qubit-n_processor-n_memory  # 辅助粒子数
n_total=n_processor*2+n_ancilla+n_memory  # 所有粒子数目

##  处理器b融合码
mapping_merge_a = np.hstack([np.arange(n_memory), np.arange(n_processor) + n_memory, np.arange(n_ancilla)+n_memory+2*n_processor])
code_merge_0a = code_merge_0.copy()
code_merge_0a.index_map(n_total,mapping_merge_a)
code_merge_1a = code_merge_1.copy()
code_merge_1a.index_map(n_total,mapping_merge_a)
code_merge_2a = code_merge_2.copy()
code_merge_2a.index_map(n_total,mapping_merge_a)
code_merge_3a = code_merge_3.copy()
code_merge_3a.index_map(n_total,mapping_merge_a)
##  处理器b融合码
mapping_merge_b=np.hstack([np.arange(n_memory),np.arange(n_processor)+n_processor+n_memory,np.arange(36)+n_processor*2+n_memory])
code_merge_0b = code_merge_0.copy()
code_merge_0b.index_map(n_total,mapping_merge_b)
code_merge_1b = code_merge_1.copy()
code_merge_1b.index_map(n_total,mapping_merge_b)
code_merge_2b = code_merge_2.copy()
code_merge_2b.index_map(n_total,mapping_merge_b)
code_merge_3b = code_merge_3.copy()
code_merge_3b.index_map(n_total,mapping_merge_b)

##  处理器a重排序
mapping_processor_a=np.arange(n_processor)+n_memory
code_processor_a=code_processor.copy()
code_processor_a.index_map(n_total,mapping_processor_a)

##  处理器b重排序
mapping_processor_b=np.arange(n_processor)+n_processor+n_memory
code_processor_b=code_processor.copy()
code_processor_b.index_map(n_total,mapping_processor_b)

stabilizers=[]
logical_n_operator_list=[]
##  内存初始化
for i in range(len(code_memory.check_list)):
    stabilizers.append(code_memory.check_list[i].copy())
for i in range(len(code_memory.logical_operator_list_x)):
    temp = code_memory.logical_operator_list_x[i].copy()
    temp.z_vector = temp.x_vector.copy()
    temp.factor = 1j
    logical_n_operator_list.append(temp.copy())
    if i == 2 or i == 3:
        temp.coff = -1
    stabilizers.append(temp)

##  处理器a初始化
for i in range(len(code_processor_a.check_list)):
    stabilizers.append(code_processor_a.check_list[i].copy())
for i in range(len(code_processor_a.logical_operator_list_x)):
    temp = code_processor_a.logical_operator_list_x[i].copy()
    temp.z_vector = temp.x_vector.copy()
    logical_n_operator_list.append(temp.copy())
    stabilizers.append(temp)

##  处理器b初始化
for i in range(len(code_processor_b.check_list)):
    stabilizers.append(code_processor_b.check_list[i].copy())
for i in range(len(code_processor_a.logical_operator_list_x)):
    temp = code_processor_b.logical_operator_list_x[i].copy()
    temp.z_vector = temp.x_vector.copy()
    logical_n_operator_list.append(temp.copy())
    stabilizers.append(temp)

##  辅助器初始化
for i in range(n_processor*2+n_memory,n_ancilla+n_memory+2*n_processor):
    op = MajoranaOperator([i], [i], 1)
    stabilizers.append(op)

#%%  SECTION：门定义
""""
input.index_0：int对象，第一个粒子序号
input.index_1：int对象，第二个粒子序号
method：str对象，作用的门的类型
"""""
def gate(computer,logical_factors,index_0,index_1,method):

    ##  纠错
    computer.loss(range(n_memory+2*n_processor))
    computer.correct(code_memory,'common')
    computer.correct(code_processor_a, 'common')
    computer.correct(code_processor_b, 'common')

    ##  选择对应的融合码
    if index_0==0 and index_1==2:
        code_temp_0=code_merge_0a
        code_temp_1=code_merge_2b
    elif index_0==0 and index_1==3:
        code_temp_0=code_merge_0a
        code_temp_1=code_merge_3b
    elif index_0==0 and index_1==1:
        code_temp_0=code_merge_0a
        code_temp_1=code_merge_1b
    elif index_0==1 and index_1==2:
        code_temp_0=code_merge_1a
        code_temp_1=code_merge_2b
    elif index_0==1 and index_1==3:
        code_temp_0=code_merge_1a
        code_temp_1=code_merge_3b
    elif index_0==2 and index_1==3:
        code_temp_0=code_merge_2a
        code_temp_1=code_merge_3b
    else:
        raise ValueError
    code_temp_0=code_temp_0.copy()
    code_temp_1=code_temp_1.copy()

    ##  将0号传输过去
    computer.correct(code_temp_0, 'relabel')
    computer.loss(range(n_memory+n_processor))
    computer.correct(code_temp_0, 'common')

    ##  根据测量结果反馈
    cache=np.append(logical_n_operator_list[index_0].x_vector,logical_n_operator_list[4].x_vector)
    cache=MajoranaOperator(cache,[],1)
    value_temp,nothing=computer.get_value(cache)
    if value_temp==-1:
        for i in logical_n_operator_list[index_0].x_vector:
            computer.phase(i)

    ##  辅助粒子解耦
    for i in range(n_memory+2*n_processor,n_ancilla+n_memory+2*n_processor):
        syn, flag = computer.measure(MajoranaOperator([i], [i], 1))
        if flag is not None and syn==-1:
            computer.stabilizers[flag].coff = 1


    ##  测量n并反馈
    syn, flag = computer.measure(logical_n_operator_list[index_0])
    if syn*logical_factors[index_0] == -1:
        logical_factors[index_0]=-logical_factors[index_0]
        logical_factors[4]=-logical_factors[4]

    ##  将1号传输过去
    computer.correct(code_temp_1, 'relabel')
    computer.loss(range(n_memory))
    computer.loss(range(n_memory+n_processor,n_memory+n_processor*2+n_ancilla))
    computer.correct(code_temp_1, 'common')

    ##  根据测量结果反馈
    cache=np.append(logical_n_operator_list[index_1].x_vector,logical_n_operator_list[5].x_vector)
    cache=MajoranaOperator(cache,[],1)
    value_temp,nothing=computer.get_value(cache)
    if value_temp==-1:
        for i in logical_n_operator_list[index_1].x_vector:
            computer.phase(i)

    ##  根据测量结果反馈
    cache=np.append(logical_n_operator_list[index_1].x_vector,logical_n_operator_list[5].x_vector)
    cache=MajoranaOperator(cache,[],1)
    value_temp,nothing=computer.get_value(cache)
    if value_temp==-1:
        for i in logical_n_operator_list[index_1].x_vector:
            computer.phase(i)

    ##  辅助解耦
    for i in range(n_memory+n_processor,n_memory+n_processor*2+n_ancilla):
        syn, flag = computer.measure(MajoranaOperator([i], [i], 1))
        if flag is not None and syn==-1:
            computer.stabilizers[flag].coff = 1


    ##  测量n并反馈
    syn, flag = computer.measure(logical_n_operator_list[index_1])
    if syn*logical_factors[index_1] == -1:
        logical_factors[index_1]=-logical_factors[index_1]
        logical_factors[5]=-logical_factors[5]

    ##  在两个处理器上施加braid门
    if method=='braid':
        for i in range(n_memory,n_memory+n_processor):
            computer.braid(i, i + n_processor)
        temp=logical_factors[4]
        logical_factors[4]=logical_factors[5]
        logical_factors[5]=temp

    ##  在两个处理器上施加fswap门
    elif method=='fswap':
        for i in range(n_memory, n_memory+n_processor):
            computer.fsawp(i, i + n_processor)
        temp=logical_factors[4]
        logical_factors[4]=logical_factors[5]
        logical_factors[5]=temp

    ##  交换回去
    computer.correct(code_temp_0, 'relabel')
    computer.loss(range(n_memory+n_processor))
    computer.correct(code_temp_0, 'common')

    ##  辅助解耦
    for i in range(n_memory+n_processor,n_memory+n_processor*2+n_ancilla):
        syn, flag = computer.measure(MajoranaOperator([i], [i], 1))
        if flag is not None and syn==-1:
            computer.stabilizers[flag].coff = 1


    ##  根据测量结果反馈
    cache=np.append(logical_n_operator_list[index_0].x_vector,logical_n_operator_list[4].x_vector)
    cache=MajoranaOperator(cache,[],1)
    value_temp,nothing=computer.get_value(cache)
    if value_temp==-1:
        for i in logical_n_operator_list[4].x_vector:
            computer.phase(i)

    ##  测量n并反馈
    syn, flag = computer.measure(logical_n_operator_list[4])
    if syn*logical_factors[4] == -1:
        logical_factors[4]=-logical_factors[4]
        logical_factors[index_0]=-logical_factors[index_0]

    ##  交换回去
    computer.correct(code_temp_1, 'relabel')
    computer.loss(range(n_memory))
    computer.loss(range(n_processor+n_memory,n_memory+n_processor*2+n_ancilla))
    computer.correct(code_temp_1, 'common')

    ##  根据测量结果反馈
    cache=np.append(logical_n_operator_list[index_1].x_vector,logical_n_operator_list[5].x_vector)
    cache=MajoranaOperator(cache,[],1)
    value_temp,nothing=computer.get_value(cache)
    if value_temp==-1:
        for i in logical_n_operator_list[5].x_vector:
            computer.phase(i)

    ##  辅助解耦
    for i in range(n_memory+n_processor,n_memory+n_processor*2+n_ancilla):
        syn, flag = computer.measure(MajoranaOperator([i], [i], 1))
        if flag is not None and syn==-1:
            computer.stabilizers[flag].coff = 1

    ##  测量n
    syn, flag = computer.measure(logical_n_operator_list[5])
    if syn*logical_factors[5] == -1:
        logical_factors[5]=-logical_factors[5]
        logical_factors[index_1]=-logical_factors[index_1]

    ##  重新进行纠错
    computer.correct(code_memory,'relabel')
    computer.correct(code_processor_a, 'relabel')
    computer.correct(code_processor_b, 'relabel')


#%%  USER：计算量子线路执行结果
def simulation(p):

    #%%  SECTION：创建不同的纠错码，初始化态矢
    """""
    态的初始化和纠错码构造
    考虑布局为：内存a，内存b，处理器a，处理器b，辅助
    """""
    ##  定义Fermionic computer
    computer=FermionicComputer()
    computer.define(n_total,p)
    ##  初始化
    logical_factors=[1,1,1,1,1,1]
    computer.initialize(stabilizers.copy())

    #%%  SECTION：进行逻辑线路
    result_list=[]
    for i in range(10):
        print(i)
        gate(computer,logical_factors,0,2,'braid')
        gate(computer,logical_factors,1,2,'braid')
        gate(computer,logical_factors,1,3,'braid')
        particle_number_0, flag = computer.get_value(logical_n_operator_list[0])
        particle_number_0=particle_number_0*logical_factors[0]
        result_list.append((1-particle_number_0)/2)
        if particle_number_0==1:
            gate(computer,logical_factors,0,2,'fswap')

        syn,flag=computer.measure(logical_n_operator_list[2])
        syn = syn * logical_factors[2]
        if syn==1:
            gate(computer,logical_factors,1,2,'fswap')
        particle_number_0, flag = computer.get_value(logical_n_operator_list[0])
        particle_number_0=particle_number_0*logical_factors[0]
        result_list.append((1-particle_number_0)/2)

        syn, flag = computer.measure(logical_n_operator_list[1])
        syn = syn * logical_factors[1]
        if syn==1:
            gate(computer,logical_factors,1,3,'fswap')
        particle_number_0, flag = computer.get_value(logical_n_operator_list[0])
        particle_number_0=particle_number_0*logical_factors[0]
        result_list.append((1-particle_number_0)/2)

    ##  返回结果
    return np.array(result_list)



if __name__ == '__main__':
    sample_number=20
    split=3
    group_number=20
    result_list=[]
    for _ in range(group_number):
        print(_)
        result_now = 0
        for i in range(split):
            with Pool(processes=sample_number) as pool:
                # 提交任务给进程池
                results = [pool.apply_async(simulation, args=(p,)) for p in [0.002]*sample_number]
                # 等待所有任务完成并获取结果
                final_results = [result.get() for result in results]
            for j in range(len(final_results)):
                result_now=result_now+final_results[j]
        result_now=result_now/(split*sample_number)
        result_list.append(result_now)
        print(result_list[-1])
    result_array=np.array(result_list)
    result_array=result_array.T
    average_array=result_array.mean(axis=1)
    std_array=result_array.std(axis=1)

    print(average_array)
    print(std_array)