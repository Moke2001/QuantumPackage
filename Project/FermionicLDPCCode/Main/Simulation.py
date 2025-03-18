import numpy as np
from matplotlib import pyplot as plt
from Physics.QuantumComputation.CliffordSimulator.FermionicComputer import FermionicComputer
from Physics.QuantumComputation.Code.ClassicalCode.EuclideanCode import EuclideanCode
from Physics.QuantumComputation.Code.QuantumCode.FermionicCode import FermionicCode
from Physics.QuantumComputation.Code.QuantumCode.FermionicLatticeSurgery import FermionicLatticeSurgery
from Physics.QuantumComputation.Code.QuantumCode.MajoranaOperator import MajoranaOperator




#%%  USER：计算量子线路执行结果
def simulation(p):

    #%%  SECTION：创建不同的纠错码，初始化态矢
    """""
    态的初始化和纠错码构造
    考虑布局为：内存a，内存b，处理器a，处理器b，辅助
    """""

    ##  创建内存纠错码
    code_memory = FermionicCode()
    code_memory.linear_combine(EuclideanCode(2, 2, 2))
    op_x_0 = MajoranaOperator([4, 6, 13, 17, 19, 21, 22],[],1)  # 内存的逻辑算符
    op_x_1 = MajoranaOperator([1, 11, 12, 14, 20, 27, 29], [], 1)  # 内存的逻辑算符
    code_memory.logical_operator_list_x=[op_x_0, op_x_1]
    code_memory.logical_operator_list_z=[op_x_0.dual(), op_x_1.dual()]
    n_memory = code_memory.number_qubit  # 内存的粒子数

    ##  创建处理器纠错码
    code_processor = FermionicCode()
    code_processor.define_qubit(7)
    code_processor.push([3, 4, 5, 6],'x')
    code_processor.push([3, 4, 5, 6],'z')
    code_processor.push([1, 2, 5, 6], 'x')
    code_processor.push([1, 2, 5, 6], 'z')
    code_processor.push([0, 2, 4, 6], 'x')
    code_processor.push([0, 2, 4, 6], 'z')
    op_x_2 = MajoranaOperator([0, 1, 2, 3, 4, 5, 6],[],1)
    code_processor.logical_operator_list_x = [op_x_2]
    code_processor.logical_operator_list_z = [op_x_2.dual()]
    n_processor = code_processor.number_qubit  # 处理器的粒子数

    ##  融合码
    code_merge_00=FermionicLatticeSurgery(code_memory, code_processor,0, 0)
    code_merge_10=FermionicLatticeSurgery(code_memory, code_processor, 1, 0)
    n_ancilla=code_merge_10.number_qubit-n_processor-n_memory  # 辅助粒子数
    n_total=n_processor*2+n_ancilla+n_memory*2  # 所有粒子数目

    ##  定义Fermionic computer
    computer=FermionicComputer()
    computer.define(n_total,p)

    ##  内存a处理器a重排序
    mapping_merge_aa=np.hstack([np.arange(30),np.arange(7)+60,np.arange(44)+30*2+7*2])

    ##  内存a处理器a融合码：第一逻辑算符
    code_merge_aa0 = code_merge_00.copy()
    code_merge_aa0.index_map(n_total,mapping_merge_aa)

    ##  内存a处理器a融合码：第二逻辑算符
    code_merge_aa1=code_merge_10.copy()
    code_merge_aa1.index_map(n_total,mapping_merge_aa)

    ##  内存a处理器b重排序
    mapping_merge_ab=np.hstack([np.arange(30),np.arange(7)+67,np.arange(44)+30*2+7*2])

    ##  内存a处理器b融合码：第一逻辑算符
    code_merge_ab0 = code_merge_00.copy()
    code_merge_ab0.index_map(n_total,mapping_merge_ab)

    ##  内存a处理器b融合码：第二逻辑算符
    code_merge_ab1=code_merge_10.copy()
    code_merge_ab1.index_map(n_total,mapping_merge_ab)

    ##  内存b处理器a重排序
    mapping_merge_ba=np.hstack([np.arange(30,60),np.arange(7)+60,np.arange(44)+30*2+7*2])

    ##  内存b处理器a融合码：第一逻辑算符
    code_merge_ba0=code_merge_00.copy()
    code_merge_ba0.index_map(n_total,mapping_merge_ba)

    ##  内存b处理器a融合码：第二逻辑算符
    code_merge_ba1=code_merge_10.copy()
    code_merge_ba1.index_map(n_total,mapping_merge_ba)

    ##  内存b处理器b重排序
    mapping_merge_bb=np.hstack([np.arange(30,60),np.arange(7)+67,np.arange(44)+30*2+7*2])

    ##  内存b处理器b融合码：第一逻辑算符
    code_merge_bb0=code_merge_00.copy()
    code_merge_bb0.index_map(n_total,mapping_merge_bb)

    ##  内存b处理器b融合码：第二逻辑算符
    code_merge_bb1=code_merge_10.copy()
    code_merge_bb1.index_map(n_total,mapping_merge_bb)

    ##  内存a重排序
    mapping_memory_a=np.arange(30)
    code_memory_a=code_memory.copy()
    code_memory_a.index_map(n_total,mapping_memory_a)

    ##  内存b重排序
    mapping_memory_b=np.arange(30,60)
    code_memory_b=code_memory.copy()
    code_memory_b.index_map(n_total,mapping_memory_b)

    ##  处理器a重排序
    mapping_processor_a=np.arange(7)+60
    code_processor_a=code_processor.copy()
    code_processor_a.index_map(n_total,mapping_processor_a)

    ##  处理器b重排序
    mapping_processor_b=np.arange(7)+67
    code_processor_b=code_processor.copy()
    code_processor_b.index_map(n_total,mapping_processor_b)

    stabilizers=[]
    logical_n_operator_list=[]

    ##  内存a初始化
    for i in range(len(code_memory_a.check_list)):
        stabilizers.append(code_memory_a.check_list[i].copy())
    for i in range(len(code_memory_a.logical_operator_list_x)):
        temp=code_memory_a.logical_operator_list_x[i].copy()
        temp.z_vector=temp.x_vector.copy()
        logical_n_operator_list.append(temp.copy())
        stabilizers.append(temp)

    ##  内存b初始化
    for i in range(len(code_memory_b.check_list)):
        stabilizers.append(code_memory_b.check_list[i].copy())
    for i in range(len(code_memory_b.logical_operator_list_x)):
        temp=code_memory_b.logical_operator_list_x[i].copy()
        temp.z_vector=temp.x_vector.copy()
        logical_n_operator_list.append(temp.copy())
        temp.coff = -1
        stabilizers.append(temp)

    ##  处理器a初始化
    for i in range(len(code_processor_a.check_list)):
        stabilizers.append(code_processor_a.check_list[i].copy())
    for i in range(len(code_processor_a.logical_operator_list_x)):
        temp=code_processor_a.logical_operator_list_x[i].copy()
        temp.z_vector=temp.x_vector.copy()
        logical_n_operator_list.append(temp.copy())
        stabilizers.append(temp)

    ##  处理器b初始化
    for i in range(len(code_processor_b.check_list)):
        stabilizers.append(code_processor_b.check_list[i].copy())
    for i in range(len(code_processor_a.logical_operator_list_x)):
        temp=code_processor_b.logical_operator_list_x[i].copy()
        temp.z_vector=temp.x_vector.copy()
        logical_n_operator_list.append(temp.copy())
        stabilizers.append(temp)

    ##  辅助器初始化
    for i in range(30*2+7*2,30*2+7*2+44):
        op=MajoranaOperator([i],[i],1)
        stabilizers.append(op)

    ##  初始化
    logical_factors=[1,1,1,1,1,1]
    computer.initialize(stabilizers)

    #%%  SECTION：门定义
    """"
    input.index_0：int对象，第一个粒子序号
    input.index_1：int对象，第二个粒子序号
    method：str对象，作用的门的类型
    """""
    def gate(index_0,index_1,method):

        ##  纠错
        computer.loss_x(range(30*2+7*2))
        computer.loss_z(range(30*2+7*2))
        computer.correct(code_memory_a,'common')
        computer.correct(code_memory_b, 'common')
        computer.correct(code_processor_a, 'common')
        computer.correct(code_processor_b, 'common')

        ##  选择对应的融合码
        if index_0==0 and index_1==2:
            code_temp_0=code_merge_aa0
            code_temp_1=code_merge_bb0
        elif index_0==0 and index_1==3:
            code_temp_0=code_merge_aa0
            code_temp_1=code_merge_bb1
        elif index_0==0 and index_1==1:
            code_temp_0=code_merge_aa0
            code_temp_1=code_merge_ab1
        elif index_0==1 and index_1==2:
            code_temp_0=code_merge_aa1
            code_temp_1=code_merge_bb0
        elif index_0==1 and index_1==3:
            code_temp_0=code_merge_aa1
            code_temp_1=code_merge_bb1
        elif index_0==2 and index_1==3:
            code_temp_0=code_merge_bb0
            code_temp_1=code_merge_bb1
        else:
            raise ValueError
        code_temp_0=code_temp_0.copy()
        code_temp_1=code_temp_1.copy()

        ##  将0号传输过去
        computer.correct(code_temp_0, 'relabel')
        if index_0==0 or  index_0==1:
            computer.loss_x(range(30))
            computer.loss_z(range(30))
        else:
            computer.loss_x(range(30,60))
            computer.loss_z(range(30,60))
        computer.loss_x(range(60,60+7))
        computer.loss_z(range(60,60+7))
        computer.correct(code_temp_0, 'common')

        ##  根据测量结果反馈
        cache=np.append(logical_n_operator_list[index_0].x_vector,logical_n_operator_list[4].x_vector)
        cache=MajoranaOperator(cache,[],1)
        value_temp,nothing=computer.get_value(cache)
        if value_temp==-1:
            for i in logical_n_operator_list[index_0].x_vector:
                computer.phase(i)

        ##  辅助粒子解耦
        for i in range(30 * 2 + 7 * 2, 30 * 2 + 7 * 2 + 44):
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
        if index_1==0 or  index_1==1:
            computer.loss_x(range(30))
            computer.loss_z(range(30))
        else:
            computer.loss_x(range(30,60))
            computer.loss_z(range(30,60))
        computer.loss_x(range(67,60+14))
        computer.loss_z(range(67,60+14))
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
        for i in range(30 * 2 + 7 * 2, 30 * 2 + 7 * 2 + 44):
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
            for i in range(30 * 2, 30 * 2 + 7):
                computer.braid(i, i + 7)
            temp=logical_factors[4]
            logical_factors[4]=logical_factors[5]
            logical_factors[5]=temp

        ##  在两个处理器上施加fswap门
        elif method=='fswap':
            for i in range(30 * 2, 30 * 2 + 7):
                computer.fsawp(i, i + 7)
            temp=logical_factors[4]
            logical_factors[4]=logical_factors[5]
            logical_factors[5]=temp

        ##  交换回去
        computer.correct(code_temp_0, 'relabel')
        if index_0==0 or  index_0==1:
            computer.loss_x(range(30))
            computer.loss_z(range(30))
        else:
            computer.loss_x(range(30,60))
            computer.loss_z(range(30,60))
        computer.loss_x(range(60,60+7))
        computer.loss_z(range(60,60+7))
        computer.correct(code_temp_0, 'common')

        ##  辅助解耦
        for i in range(30 * 2 + 7 * 2, 30 * 2 + 7 * 2 + 44):
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
        if index_1 == 0 or index_1 == 1:
            computer.loss_x(range(30))
            computer.loss_z(range(30))
        else:
            computer.loss_x(range(30, 60))
            computer.loss_z(range(30, 60))
        computer.loss_x(range(67, 60 + 14))
        computer.loss_z(range(67, 60 + 14))
        computer.correct(code_temp_1, 'common')

        ##  根据测量结果反馈
        cache=np.append(logical_n_operator_list[index_1].x_vector,logical_n_operator_list[5].x_vector)
        cache=MajoranaOperator(cache,[],1)
        value_temp,nothing=computer.get_value(cache)
        if value_temp==-1:
            for i in logical_n_operator_list[5].x_vector:
                computer.phase(i)

        ##  辅助解耦
        for i in range(30 * 2 + 7 * 2, 30 * 2 + 7 * 2 + 44):
            syn, flag = computer.measure(MajoranaOperator([i], [i], 1))
            if flag is not None and syn==-1:
                computer.stabilizers[flag].coff = 1

        ##  测量n
        syn, flag = computer.measure(logical_n_operator_list[5])
        if syn*logical_factors[5] == -1:
            logical_factors[5]=-logical_factors[5]
            logical_factors[index_1]=-logical_factors[index_1]

        ##  重新进行纠错
        computer.correct(code_memory_a,'common')
        computer.correct(code_memory_b, 'common')
        computer.correct(code_processor_a, 'common')
        computer.correct(code_processor_b, 'common')


    #%%  SECTION：进行逻辑线路
    for i in range(30):
        print(i)
        gate(0,2,'braid')
        gate(1,2,'braid')
        gate(1,3,'braid')
        print(computer.noise_recoder_x)
        print(computer.noise_recoder_z)
        particle_number_0, flag = computer.measure(logical_n_operator_list[0])
        particle_number_0=particle_number_0*logical_factors[0]
        print((1-particle_number_0)/2)
        if particle_number_0==1:
            gate(0,2,'fswap')

        syn,flag=computer.measure(logical_n_operator_list[2])
        syn = syn * logical_factors[2]
        if syn==1:
            gate(1,2,'fswap')
        print(computer.noise_recoder_x)
        print(computer.noise_recoder_z)
        particle_number_0,nothing = computer.get_value(logical_n_operator_list[0])
        particle_number_0 = particle_number_0 * logical_factors[0]
        print((1 - particle_number_0) / 2)

        syn, flag = computer.measure(logical_n_operator_list[1])
        syn = syn * logical_factors[1]
        if syn==1:
            gate(1,3,'fswap')
        print(computer.noise_recoder_x)
        print(computer.noise_recoder_z)
        particle_number_0,nothing = computer.get_value(logical_n_operator_list[0])
        particle_number_0 = particle_number_0 * logical_factors[0]
        print((1 - particle_number_0) / 2)

    ##  返回结果
    return None



if __name__ == '__main__':
    # 创建进程池
    # sample_number=100
    # with Pool(processes=50) as pool:
    #     # 提交任务给进程池
    #     results = [pool.apply_async(simulation, args=(p,)) for p in [0.01]*sample_number]
    #
    #     # 等待所有任务完成并获取结果
    #     final_results = [result.get() for result in results]

    # # 打印结果
    # result=0
    # for result_temp in final_results:
    #     result += result_temp
    # result /= sample_number
    result=simulation(0.01)
    plt.plot(range(len(result)), result)
    plt.show()