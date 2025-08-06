import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from MokeQuantumComputation.Code.QuantumCode.Fermion.FermionicCode import FermionicCode
from MokeQuantumComputation.Code.QuantumCode.Fermion.MajoranaOperator import MajoranaOperator
from MokeQuantumComputation.Helper.CycleComposition import GreedyCycleDecomposition


#%%  USER：生成Fermionic lattice surgery之后的code
"""
input.code_0：FermionicCode对象
input.code_1：FermionicCode对象
input.index_0：int对象，逻辑算符下标
input.index_1：int对象，逻辑算符下标
ouput：FermionicCode对象，融合结果
"""

def FermionicLatticeSurgery(code_A, code_B, index_A, index_B)->FermionicCode:


    #%%  SECTION：数据标准化
    code_A=code_A.copy()
    code_B=code_B.copy()
    assert isinstance(code_A, FermionicCode)
    assert isinstance(code_B, FermionicCode)
    assert isinstance(index_A, int)
    assert isinstance(index_B, int)


    #%%  SECTION：数据预处理
    number_qubit_A=code_A.number_qubit
    number_qubit_B=code_B.number_qubit
    code_B.index_map(number_qubit_A+number_qubit_B,np.arange(number_qubit_B)+number_qubit_A)
    code_A.index_map(number_qubit_A+number_qubit_B,np.arange(number_qubit_A))
    logical_operator_0=code_A.logical_operator_list_x[index_A]
    logical_operator_1=code_B.logical_operator_list_x[index_B]
    support_index_vector_A=np.array(logical_operator_0.x_vector,dtype=int)
    support_index_vector_B=np.array(logical_operator_1.x_vector,dtype=int)

    ##  提取与这些费米子相关联的校验子
    check_list_x_fix=[]
    check_list_x_unfix=[]
    check_list_z=[]
    check_list_new=[]
    vertex_qubit_list=[]
    vertex_check_list=[]
    edge_list=[]
    for i in range(len(code_A.check_list)):
        if len(code_A.check_list[i].x_vector)>0:
            if len(set(code_A.check_list[i].x_vector)&set(support_index_vector_A))>0:
                check_list_x_fix.append(code_A.check_list[i].copy())
            else:
                check_list_x_unfix.append(code_A.check_list[i].copy())
        else:
            check_list_z.append(code_A.check_list[i].copy())

    for i in range(len(code_B.check_list)):
        if len(code_B.check_list[i].x_vector)>0:
            if len(set(code_B.check_list[i].x_vector)&set(support_index_vector_B))>0:
                check_list_x_fix.append(code_B.check_list[i].copy())
            else:
                check_list_x_unfix.append(code_B.check_list[i].copy())
        else:
            check_list_z.append(code_B.check_list[i].copy())

    ##  为待修改的stabilizers增加ancilla及其索引
    code=FermionicCode()
    code.define_qubit(number_qubit_A+number_qubit_B)
    ancilla_list_list=[]
    for i in range(len(check_list_x_fix)):
        x_vector_temp=check_list_x_fix[i].x_vector.tolist()
        z_vector_temp=[]
        if check_list_x_fix[i].x_vector[0]<number_qubit_A:
            vertex_check_list.append((str(len(vertex_check_list))+'A'))
        else:
            vertex_check_list.append((str(len(vertex_check_list)) + 'B'))
        number_ancilla_temp=len(set(check_list_x_fix[i].x_vector)&set(np.append(support_index_vector_A,support_index_vector_B)))//2
        temp=[]
        for j in range(number_ancilla_temp):
            code.push_qubit(1)
            temp.append((code.qubit_list[-1],'x'))
            temp.append((code.qubit_list[-1],'z'))
            x_vector_temp.append(code.qubit_list[-1])
            z_vector_temp.append(code.qubit_list[-1])
            edge_list.append((str(code.qubit_list[-1])+'x',vertex_check_list[-1]))
            edge_list.append((str(code.qubit_list[-1]) + 'z', vertex_check_list[-1]))
        check_list_x_fix[i]=MajoranaOperator(x_vector_temp,z_vector_temp,1)
        ancilla_list_list.append(temp)
    for i in range(number_qubit_A+number_qubit_B,code.number_qubit):
        vertex_qubit_list.append(str(i)+'x')
        vertex_qubit_list.append(str(i)+'z')


    #%%  SECTION：加入测量稳定子
    single_point=None
    single_qubit_list=[]
    ##  右边logical更长的情况
    if len(support_index_vector_A)<len(support_index_vector_B):

        ##  先将两边对齐的部分连起来
        for i in range(len(support_index_vector_A)):
            vertex_check_list.append((str(len(vertex_check_list)) + 'M'))
            x_vector_temp=[support_index_vector_A[i],support_index_vector_B[i]]
            z_vector_temp=[]
            for j in range(len(check_list_x_fix)):
                if support_index_vector_A[i] in check_list_x_fix[j].x_vector or support_index_vector_B[i] in check_list_x_fix[j].x_vector:
                    temp=ancilla_list_list[j][-1]
                    if temp[1]=='x':
                        x_vector_temp.append(temp[0])
                        edge_list.append((str(temp[0])+'x',vertex_check_list[-1]))
                    elif temp[1]=='z':
                        z_vector_temp.append(temp[0])
                        edge_list.append((str(temp[0]) + 'z', vertex_check_list[-1]))
                    ancilla_list_list[j].pop(-1)

            if np.mod(len(x_vector_temp)+len(z_vector_temp),2)!=0:
                if single_point is None:
                    code.push_qubit(1)
                    single_point=code.qubit_list[-1]
                    vertex_qubit_list.append(str(single_point)+'x')
                    vertex_qubit_list.append(str(single_point)+'z')
                    edge_list.append((str(single_point)+'x',vertex_check_list[-1]))
                    single_qubit_list.append(code.qubit_list[-1])
                    x_vector_temp.append(code.qubit_list[-1])
                else:
                    z_vector_temp.append(single_point)
                    edge_list.append((str(single_point) + 'z', vertex_check_list[-1]))
                    single_point=None

            ##  引入新的measurement stabilizer
            check_list_new.append(MajoranaOperator(x_vector_temp,z_vector_temp,1))

        ##  将右边剩余的部分连起来
        length_B=len(support_index_vector_B)
        length_A=len(support_index_vector_A)
        for i in range((length_B-length_A)//2):
            index_0=length_A+2*i
            index_1=index_0+1
            vertex_check_list.append((str(len(vertex_check_list)) + 'M'))
            x_vector_temp = [support_index_vector_B[index_0], support_index_vector_B[index_1]]
            z_vector_temp = []
            for j in range(len(check_list_x_fix)):
                if support_index_vector_B[index_0] in check_list_x_fix[j].x_vector or support_index_vector_B[index_1] in check_list_x_fix[j].x_vector:
                    temp = ancilla_list_list[j][-1]
                    if temp[1]=='x':
                        x_vector_temp.append(temp[0])
                        edge_list.append((str(temp[0])+'x',vertex_check_list[-1]))
                    elif temp[1]=='z':
                        z_vector_temp.append(temp[0])
                        edge_list.append((str(temp[0]) + 'z', vertex_check_list[-1]))
                    ancilla_list_list[j].pop(-1)

            if np.mod(len(x_vector_temp) + len(z_vector_temp), 2) != 0:
                if single_point is None:
                    code.push_qubit(1)
                    single_point=code.qubit_list[-1]
                    vertex_qubit_list.append(str(single_point)+'x')
                    vertex_qubit_list.append(str(single_point)+'z')
                    edge_list.append((str(single_point)+'x',vertex_check_list[-1]))
                    single_qubit_list.append(code.qubit_list[-1])
                    x_vector_temp.append(code.qubit_list[-1])
                else:
                    z_vector_temp.append(single_point)
                    edge_list.append((str(single_point) + 'z', vertex_check_list[-1]))
                    single_point=None

            ##  引入新的measurement stabilizer
            check_list_new.append(MajoranaOperator(x_vector_temp, z_vector_temp, 1))


    ##  左边logical更长的情况
    else:

        ##  先将两边对齐的部分连起来
        for i in range(len(support_index_vector_B)):
            vertex_check_list.append((str(len(vertex_check_list)) + 'M'))
            x_vector_temp = [support_index_vector_A[i], support_index_vector_B[i]]
            z_vector_temp = []
            for j in range(len(check_list_x_fix)):
                if support_index_vector_A[i] in check_list_x_fix[j].x_vector or support_index_vector_B[i] in check_list_x_fix[j].x_vector:
                    temp = ancilla_list_list[j][-1]
                    if temp[1]=='x':
                        x_vector_temp.append(temp[0])
                        edge_list.append((str(temp[0])+'x',vertex_check_list[-1]))
                    elif temp[1]=='z':
                        z_vector_temp.append(temp[0])
                        edge_list.append((str(temp[0]) + 'z', vertex_check_list[-1]))
                    ancilla_list_list[j].pop(-1)

            if np.mod(len(x_vector_temp) + len(z_vector_temp), 2) != 0:
                if single_point is None:
                    code.push_qubit(1)
                    single_point=code.qubit_list[-1]
                    vertex_qubit_list.append(str(single_point)+'x')
                    vertex_qubit_list.append(str(single_point)+'z')
                    edge_list.append((str(single_point)+'x',vertex_check_list[-1]))
                    single_qubit_list.append(code.qubit_list[-1])
                    x_vector_temp.append(code.qubit_list[-1])
                else:
                    z_vector_temp.append(single_point)
                    edge_list.append((str(single_point) + 'z', vertex_check_list[-1]))
                    single_point=None

            ##  引入新的measurement stabilizer
            check_list_new.append(MajoranaOperator(x_vector_temp, z_vector_temp, 1))

        ##  将右边剩余的部分连起来
        length_B=len(support_index_vector_B)
        length_A=len(support_index_vector_A)
        for i in range((length_A-length_B)//2):
            index_0=length_B+2*i
            index_1=index_0+1
            vertex_check_list.append((str(len(vertex_check_list)) + 'M'))
            x_vector_temp = [support_index_vector_A[index_0], support_index_vector_A[index_1]]
            z_vector_temp = []
            for j in range(len(check_list_x_fix)):
                if support_index_vector_A[index_0] in check_list_x_fix[j].x_vector:
                    temp = ancilla_list_list[j][-1]
                    if temp[1]=='x':
                        x_vector_temp.append(temp[0])
                        edge_list.append((str(temp[0])+'x',vertex_check_list[-1]))
                    elif temp[1]=='z':
                        z_vector_temp.append(temp[0])
                        edge_list.append((str(temp[0]) + 'z', vertex_check_list[-1]))
                    ancilla_list_list[j].pop(-1)
                if support_index_vector_A[index_1] in check_list_x_fix[j].x_vector:
                    temp = ancilla_list_list[j][-1]
                    if temp[1]=='x':
                        x_vector_temp.append(temp[0])
                        edge_list.append((str(temp[0])+'x',vertex_check_list[-1]))
                    elif temp[1]=='z':
                        z_vector_temp.append(temp[0])
                        edge_list.append((str(temp[0]) + 'z', vertex_check_list[-1]))
                    ancilla_list_list[j].pop(-1)

            if np.mod(len(x_vector_temp) + len(z_vector_temp), 2) != 0:
                if single_point is None:
                    code.push_qubit(1)
                    single_point=code.qubit_list[-1]
                    vertex_qubit_list.append(str(single_point)+'x')
                    vertex_qubit_list.append(str(single_point)+'z')
                    edge_list.append((str(single_point)+'x',vertex_check_list[-1]))
                    single_qubit_list.append(code.qubit_list[-1])
                    x_vector_temp.append(code.qubit_list[-1])
                else:
                    z_vector_temp.append(single_point)
                    edge_list.append((str(single_point) + 'z', vertex_check_list[-1]))
                    single_point=None

            ##  引入新的measurement stabilizer
            check_list_new.append(MajoranaOperator(x_vector_temp, z_vector_temp, 1))


    #%%  SECTION：图论计算规范稳定子

    ##  获取关键参数

    lst=vertex_check_list

    rst=vertex_qubit_list

    vertex_check_list.append('D')
    for i in range(len(single_qubit_list)):
        edge_list.append((str(single_qubit_list[i])+'x','D'))
        edge_list.append((str(single_qubit_list[i])+'z', 'D'))

    ##  求最小不交圈的覆盖
    node_measurement_list=[]
    node_origin_list=[]
    node_qubit_list=[]
    node_qubit_colors = []
    node_origin_colors=[]
    node_measurement_colors=[]
    for node in vertex_check_list+vertex_qubit_list:
        if node[-1]=='x' or node[-1]=='z':
            node_qubit_list.append(node)
            node_qubit_colors.append('red')
        elif node[-1]=='A':
            node_origin_list.append(node)
            node_origin_colors.append('blue')
        elif node[-1]=='B':
            node_origin_list.append(node)
            node_origin_colors.append('green')
        elif node[-1]=='M':
            node_measurement_list.append(node)
            node_measurement_colors.append('yellow')
        elif node[-1]=='D':
            node_measurement_list.append(node)
            node_measurement_colors.append('cyan')
    node_colors=node_qubit_colors+node_origin_colors+node_measurement_colors
    graph=nx.Graph()
    graph.add_nodes_from(node_qubit_list)
    graph.add_nodes_from(node_origin_list)
    graph.add_nodes_from(node_measurement_list)
    graph.add_edges_from(edge_list)
    nx.draw(graph,pos=nx.kamada_kawai_layout(graph),node_color=node_colors)
    plt.savefig('pic.pdf')
    number_measurement_stabilizer=len(check_list_new)
    number_ancilla=len(vertex_qubit_list)//2
    composition=GreedyCycleDecomposition(graph)
    cycle_list = []
    for i in range(len(composition)):
        temp=[]
        for j in range(len(composition[i])):
            if isinstance(composition[i][j],str) and composition[i][j]!='A':
                temp.append(composition[i][j])
        cycle_list.append(np.unique(temp))

    ##  将不交圈转换为gauge stabilizers
    check_gauge=[]
    for i in range(len(cycle_list)):
        x_vector_temp = []
        z_vector_temp = []
        for j in range(len(cycle_list[i])):
            if cycle_list[i][j][-1]=='x':
                x_vector_temp.append(int(cycle_list[i][j][0:-1]))
            elif cycle_list[i][j][-1]=='z':
                z_vector_temp.append(int(cycle_list[i][j][0:-1]))
        check_gauge.append(MajoranaOperator(x_vector_temp, z_vector_temp, 1))

    code.check_list=check_list_x_unfix+check_list_x_fix+check_list_z+check_gauge+check_list_new
    code.number_checker=len(code.check_list)


    #%%  SECTION：返回结果

    assert code.commute_judge()  # 检查对易性
    return code