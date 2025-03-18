import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from Physics.QuantumComputation.Code.ClassicalCode.EuclideanCode import EuclideanCode
from Physics.QuantumComputation.Code.QuantumCode.FermionicCode import FermionicCode
from Physics.QuantumComputation.Code.QuantumCode.MajoranaOperator import MajoranaOperator


#%%  USER：生成Fermionic lattice surgery之后的code
def FermionicLatticeSurgery(code_0, code_1, index_0, index_1)->FermionicCode:

    ##  数据标准化
    code_0=code_0.copy()
    code_1=code_1.copy()
    assert isinstance(code_0, FermionicCode)
    assert isinstance(code_1, FermionicCode)
    assert isinstance(index_0, int)
    assert isinstance(index_1, int)

    ##  获取逻辑算符
    number_qubit_0=code_0.number_qubit
    number_qubit_1=code_1.number_qubit
    check_list_x_0,check_list_z_0=code_0.css()
    check_list_x_1,check_list_z_1=code_1.css()
    number_checker_x_0=len(check_list_x_0)
    number_checker_z_0=len(check_list_z_0)
    for i in range(len(check_list_x_1)):
        check_list_x_1[i].shift(number_qubit_0)
    for j in range(len(check_list_z_1)):
        check_list_z_1[j].shift(number_qubit_0)
    number_checker_x_1=len(check_list_x_1)
    number_checker_z_1=len(check_list_z_1)
    logical_0=code_0.get_logical_operators()[0][index_0].x_vector
    logical_1=code_1.get_logical_operators()[0][index_1].x_vector

    ##  提取与这些费米子相关联的校验子
    ass_list_0=[]
    ass_list_1=[]
    for i in range(logical_0.shape[0]):
        temp=[]
        for j in range(number_checker_x_0):
            if np.any(check_list_x_0[j].x_vector==logical_0[i]):
                temp.append(j)
        ass_list_0.append(np.array(temp,dtype=int))
    for i in range(logical_1.shape[0]):
        temp=[]
        for j in range(number_checker_x_1):
            if np.any(check_list_x_1[j].x_vector==logical_1[i]+number_qubit_0):
                temp.append(j)
        ass_list_1.append(np.array(temp,dtype=int))

    ##  加入原先已有的校验子
    code_merge=FermionicCode()  # 结果初始化
    code_merge.define_qubit(number_qubit_0+number_qubit_1)
    x_list=check_list_x_0+check_list_x_1
    z_list=check_list_z_0+check_list_z_1

    ##  右边logical更长的情况
    if len(logical_0)<len(logical_1):

        ##  先将两边对齐的部分连起来
        for i in range(logical_0.shape[0]):
            temp=[logical_0[i],logical_1[i]+code_0.number_qubit]  # measurement stabilizers初始化

            ##  与左边的qubit关联的original stabilizers的处理
            for j in range(len(ass_list_0[i])):
                code_merge.push_qubit(1)  # 添加一个ancilla

                ##  修改原先的stabilizer
                x_vector=np.append(x_list[ass_list_0[i][j]].x_vector,code_merge.qubit_list[-1])
                z_vector=np.append(x_list[ass_list_0[i][j]].z_vector,code_merge.qubit_list[-1])
                x_list[ass_list_0[i][j]]=MajoranaOperator(x_vector,z_vector,1)

                ##  将measurement stabilizer与ancilla相连
                temp.append(code_merge.qubit_list[-1])

            ##  与右边的qubit关联的original stabilizers的处理
            for j in range(len(ass_list_1[i])):
                code_merge.push_qubit(1)  # 添加一个ancilla

                ##  修改原先的stabilizer
                x_vector=np.append(x_list[ass_list_1[i][j].x_vector+number_checker_x_0],code_merge.qubit_list[-1])
                z_vector=np.append(x_list[ass_list_1[i][j].z_vector+number_checker_x_0],code_merge.qubit_list[-1])
                x_list[ass_list_1[i][j]]=MajoranaOperator(x_vector,z_vector,1)

                ##  将measurement stabilizer与ancilla相连
                temp.append(code_merge.qubit_list[-1])

            ##  如果measurement stabilizers的权重是奇数，需要再加上一个ancilla
            if np.mod(len(temp),2)!=0:
                code_merge.push_qubit(1)
                temp.append(code_merge.qubit_list[-1])

            ##  引入新的measurement stabilizer
            x_list.append(MajoranaOperator(temp,[],1))

        ##  将右边剩余的部分连起来
        for i in range((logical_1.shape[0]-logical_0.shape[0])//2):
            ##  measurement stabilizers初始化
            index__0=logical_1[2*i+logical_0.shape[0]]+code_0.number_qubit
            index__1=logical_1[2*i+1+logical_0.shape[0]]+code_0.number_qubit
            temp=[index__0,index__1]

            ##  与qubit关联的original stabilizers的处理
            for j in range(len(ass_list_1[2*i+logical_0.shape[0]])):
                code_merge.push_qubit(1)
                x_vector=np.append(x_list[ass_list_1[i][j]+number_checker_x_0].x_vector,code_merge.qubit_list[-1])
                z_vector=np.append(x_list[ass_list_1[i][j]+number_checker_x_0].z_vector,code_merge.qubit_list[-1])
                x_list[ass_list_1[i][j]+number_checker_x_0]=MajoranaOperator(x_vector,z_vector,1)
                temp.append(code_merge.qubit_list[-1])
            for j in range(len(ass_list_1[2*i+1+logical_0.shape[0]])):
                code_merge.push_qubit(1)
                x_vector=np.append(x_list[ass_list_1[i][j]+number_checker_x_0].x_vector,code_merge.qubit_list[-1])
                z_vector=np.append(x_list[ass_list_1[i][j]+number_checker_x_0].z_vector,code_merge.qubit_list[-1])
                x_list[ass_list_1[i][j]+number_checker_x_0]=MajoranaOperator(x_vector,z_vector,1)
                temp.append(code_merge.qubit_list[-1])

            ##  如果measurement stabilizers的权重是奇数，需要再加上一个ancilla
            if np.mod(len(temp),2)!=0:
                code_merge.push_qubit(1)
                temp.append(code_merge.qubit_list[-1])

            ##  引入新的measurement stabilizer
            x_list.append(MajoranaOperator(temp, [], 1))

    ##  左边logical更长的情况
    else:

        ##  先将两边对齐的部分连起来
        for i in range(len(logical_1)):
            temp=[logical_0[i],logical_1[i]+code_0.number_qubit]  # measurement stabilizers初始化

            ##  与左边的qubit关联的original stabilizers的处理
            for j in range(len(ass_list_0[i])):
                code_merge.push_qubit(1)  # 添加一个ancilla

                ##  修改原先的stabilizer
                x_vector = np.append(x_list[ass_list_0[i][j]].x_vector, code_merge.qubit_list[-1])
                z_vector = np.append(x_list[ass_list_0[i][j]].z_vector,code_merge.qubit_list[-1])
                x_list[ass_list_0[i][j]]=MajoranaOperator(x_vector, z_vector, 1)

                ##  将measurement stabilizer与ancilla相连
                temp.append(code_merge.qubit_list[-1])

            ##  与右边的qubit关联的original stabilizers的处理
            for j in range(len(ass_list_1[i])):
                code_merge.push_qubit(1)  # 添加一个ancilla

                ##  修改原先的stabilizer
                x_vector = np.append(x_list[ass_list_1[i][j] + number_checker_x_0].x_vector, code_merge.qubit_list[-1])
                z_vector = np.append(x_list[ass_list_1[i][j] + number_checker_x_0].z_vector,code_merge.qubit_list[-1])
                x_list[ass_list_1[i][j] + number_checker_x_0]=MajoranaOperator(x_vector, z_vector, 1)

                ##  将measurement stabilizer与ancilla相连
                temp.append(code_merge.qubit_list[-1])

            ##  如果measurement stabilizers的权重是奇数，需要再加上一个ancilla
            if np.mod(len(temp), 2) != 0:
                code_merge.push_qubit(1)
                temp.append(code_merge.qubit_list[-1])

            ##  引入新的measurement stabilizer
            x_list.append(MajoranaOperator(temp, [], 1))

        ##  将左边剩余的部分连起来
        for i in range((len(logical_0)-len(logical_1))//2):

            ##  measurement stabilizers初始化
            index__0=logical_0[2*i+len(logical_1)]
            index__1=logical_0[2*i+1+len(logical_1)]
            temp=[index__0,index__1]

            ##  与qubit关联的original stabilizers的处理
            for j in range(len(ass_list_0[2*i+len(logical_1)])):
                code_merge.push_qubit(1)
                x_vector = np.append(x_list[ass_list_0[i][j]].x_vector, code_merge.qubit_list[-1])
                z_vector = np.append(x_list[ass_list_0[i][j]].z_vector,code_merge.qubit_list[-1])
                x_list[ass_list_0[i][j]]=MajoranaOperator(x_vector, z_vector, 1)
                temp.append(code_merge.qubit_list[-1])
            for j in range(len(ass_list_0[2*i+1+logical_1.shape[0]])):
                code_merge.push_qubit(1)
                x_vector = np.append(x_list[ass_list_0[i][j]].x_vector, code_merge.qubit_list[-1])
                z_vector = np.append(x_list[ass_list_0[i][j]].x_vector,code_merge.qubit_list[-1])
                x_list[ass_list_0[i][j]]=MajoranaOperator(x_vector, z_vector, 1)
                temp.append(code_merge.qubit_list[-1])

            ##  如果measurement stabilizers的权重是奇数，需要再加上一个ancilla
            if np.mod(len(temp),2)!=0:
                code_merge.push_qubit(1)
                temp.append(code_merge.qubit_list[-1])

            ##  引入新的measurement stabilizer
            x_list.append(MajoranaOperator(temp, [], 1))

    ##  获取关键参数
    number_vertice_qubit=code_merge.number_qubit-code_0.number_qubit-code_1.number_qubit  # 当前ancilla的数目
    number_origin = code_0.number_qubit + code_1.number_qubit  # 过去qubit的数目
    lst=range(number_origin,number_vertice_qubit+number_origin)  # 二部图的ancilla顶点
    rst=[]  # 二部图的stabilizer顶点
    edge_list=[]  # 二部图的边
    for i in range(len(x_list)):
        code_merge.push(x_list[i])
    for i in range(len(z_list)):
        code_merge.push(z_list[i])

    ##  求无向二部图
    for i in range(code_merge.number_checker):
        overlap=[temp for temp in code_merge.check_list[i].x_vector if temp>=number_origin]
        if len(overlap)!=0:
            rst.append(-i)
        for j in range(len(overlap)):
            edge_list.append((-i,overlap[j]))

    ##  求最小不交圈的覆盖
    graph=nx.Graph()
    graph.add_nodes_from(lst)
    graph.add_nodes_from(rst)
    graph.add_edges_from(edge_list)
    # nx.draw(graph, pos=nx.kamada_kawai_layout(graph),node_size=500,with_labels=True, node_color=['g'] * len(lst) + ['r'] * len(rst))
    # plt.savefig('test.pdf')
    if index_0==0:
        cycle_list=[[49,52,61,64],
                    [45,47,50,53],
                    [42,37,43,48],
                    [73,75,46,44],
                    [76,40,39,51,54,79],
                    [71,77,78,72],
                    [63,69,68,38,41,66],
                    [58,70,67,55],
                    [60,80,74,56],
                    [57,62,65,59],
                    ]
    elif index_0==1:
        cycle_list=[[62,61,55,56],
                    [60,58,39,42],
                    [38,37,50,49],
                    [78,76,53,52],
                    [79,74,44,46,40,41],
                    [80,73,68,69,45,48],
                    [64,63,75,77,71,70],
                    [67,72,47,43],
                    [65,66,54,51,57,59]
                    ]
    else:
        raise IndexError
    ##  将不交圈转换为gauge stabilizers
    for i in range(len(cycle_list)):
        code_merge.push(MajoranaOperator(cycle_list[i],[],1))

    ##  检查对易性
    assert code_merge.commute_judge()

    ##  返回结果
    return code_merge


if __name__ == '__main__':
    code_0 = FermionicCode()
    code_0.linear_combine(EuclideanCode(2, 2, 2))
    op_x_0 = MajoranaOperator([4, 6, 13, 17, 19, 21, 22],[],1)
    op_x_1 = MajoranaOperator([1, 11, 12, 14, 20, 27, 29], [], 1)
    code_0.logical_operator_list_x=[op_x_0, op_x_1]
    code_0.logical_operator_list_z=[op_x_0.dual(), op_x_1.dual()]
    n_0 = code_0.number_qubit

    ##  定义processor code：Fermionic Steane code
    code_1 = FermionicCode()
    code_1.define_qubit(7)
    code_1.push([3, 4, 5, 6],'x')
    code_1.push([3, 4, 5, 6],'z')
    code_1.push([1, 2, 5, 6], 'x')
    code_1.push([1, 2, 5, 6], 'z')
    code_1.push([0, 2, 4, 6], 'x')
    code_1.push([0, 2, 4, 6], 'z')
    op_x_2 = MajoranaOperator([0, 1, 2, 3, 4, 5, 6],[],1)
    code_1.logical_operator_list_x = [op_x_2]
    code_1.logical_operator_list_z = [op_x_2.dual()]
    FermionicLatticeSurgery(code_0, code_1, 1,0)
    pass