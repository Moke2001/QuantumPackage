import numpy as np
from MokeQuantumSimulation.Algorithm.Interface.InterfaceQutip.FunctionQutip import FunctionQutip
from MokeQuantumSimulation.Format.ModelFormat.ModelFormat import ModelFormat
from MokeQuantumSimulation.Format.TermFormat.CouplingTermFormat import CouplingTermFormat
from MokeQuantumSimulation.Format.TermFormat.MultiTermFormat import MultiTermFormat
from MokeQuantumSimulation.Format.TermFormat.OnsiteTermFormat import OnsiteTermFormat
from MokeQuantumSimulation.Format.TermFormat.OverallCouplingTermFormat import OverallCouplingTermFormat
from MokeQuantumSimulation.Format.TermFormat.OverallMultiTermFormat import OverallMultiTermFormat
from MokeQuantumSimulation.Format.TermFormat.OverallOnsiteTermFormat import OverallOnsiteTermFormat
from MokeQuantumSimulation.Format.TermFormat.TermFormat import TermFormat
from MokeQuantumSimulation.Algorithm.Interface.InterfaceQutip.GetLocalOperatorQutip import get_local_operator_qutip


#%%  KEY：根据格式作用量返回qutip算符
"""
input.model：ModelFormat对象，模型
input.term：TermFormat对象，单一格式作用量
output：Qobj对象，对应于系统的qutip形式算符
influence：本函数不改变参数对象
"""
def get_operator_qutip(model,term):
    ##  SECTION：标准化-----------------------------------------------------------------------------
    assert isinstance(term, TermFormat), '参数term必须是TermFormat对象'
    assert isinstance(model, ModelFormat),'参数model必须是ModelFormat对象'
    qobj = 0

    ##  SECTION：OnsiteTermFormat作用量-------------------------------------------------------------
    if isinstance(term, OnsiteTermFormat):
        qobj = get_local_operator_qutip(model,term.get_position(), term.get_op())

    ##  SECTION：CouplingTermFormat作用量-----------------------------------------------------------
    elif isinstance(term, CouplingTermFormat):
        op_0, op_1 = term.get_op()  # 两个算符名称
        index_tuple_0, index_tuple_1 = term.get_position()  # 两个位点
        qobj = get_local_operator_qutip(model,index_tuple_0, op_0) * get_local_operator_qutip(model,index_tuple_1, op_1)

    ##  SECTION：MultiTermFormat作用量--------------------------------------------------------------
    elif isinstance(term, MultiTermFormat):
        op_list = term.get_op()  # 算符名称列表
        index_tuple_list = term.get_position()  # 位点列表
        qobj = 1  # 结果初始化
        for i in range(len(op_list)):
            qobj = qobj * get_local_operator_qutip(model,index_tuple_list[i], op_list[i])

    ##  SECTION：OverallOnsiteTermFormat作用量------------------------------------------------------
    elif isinstance(term, OverallOnsiteTermFormat):
        op = term.get_op()  # 算符名称
        inner_index = term.get_position()  # 胞内序号
        ##  对晶格遍历，发现胞内坐标符合的就停下来求和
        for index_tuple, site_iteration in np.ndenumerate(model.dimension_array):
            if index_tuple[-1] == inner_index:
                qobj = qobj + get_local_operator_qutip(model,index_tuple, op)

    ##  SECTION：OverallCouplingTermFormat作用量----------------------------------------------------
    elif isinstance(term, OverallCouplingTermFormat):
        op_0, op_1 = term.get_op()  # 算符名称
        inner_index_0, inner_index_1, cell_vector = term.get_position()  # 胞内序号和相对格位移

        ##  遍历晶格
        for index_tuple_iteration, site_iteration in np.ndenumerate(model.dimension_array[..., 0]):
            index_tuple_0_temp = index_tuple_iteration + (inner_index_0,)  # 第一个作用位点

            ##  第二个作用位点
            index_tuple_1_temp_temp = ()
            for i in range(len(cell_vector)):
                index_tuple_1_temp_temp = index_tuple_1_temp_temp + (cell_vector[i] + index_tuple_0_temp[i],)
            index_tuple_1_temp_temp = index_tuple_1_temp_temp + (inner_index_1,)

            ##  应用周期条件
            bc=model.periodicity
            index_tuple_1_temp=()
            if isinstance(bc,bool):
                if bc:
                    for i in range(len(index_tuple_1_temp_temp)-1):
                        index_tuple_1_temp=index_tuple_1_temp+(np.mod(index_tuple_1_temp_temp[i],model.cell_period_list[i]),)
                    index_tuple_1_temp = index_tuple_1_temp + (index_tuple_1_temp_temp[-1],)
                else:
                    index_tuple_1_temp=index_tuple_1_temp_temp
            elif isinstance(bc,list):
                for i in range(len(index_tuple_1_temp_temp) - 1):
                    if bc[i]:
                        index_tuple_1_temp = index_tuple_1_temp + (np.mod(index_tuple_1_temp_temp[i], model.cell_period_list[i]),)
                index_tuple_1_temp = index_tuple_1_temp + (index_tuple_1_temp_temp[-1],)

            ##  判断位点不要跳出晶格范围
            if all(index_tuple_1_temp[it] < model.cell_period_list[it] for it in range(len(model.cell_period_list))):
                qobj = qobj + get_local_operator_qutip(model,index_tuple_0_temp, op_0) * get_local_operator_qutip(model,index_tuple_1_temp, op_1)

    ##  SECTION：OverallMultiTermFormat作用量-------------------------------------------------------
    elif isinstance(term, OverallMultiTermFormat):
        op_list = term.get_op()  # 算符名称
        inner_index_list, cell_vector_list = term.get_position()  # 胞内序号和相对格位移

        ##  遍历晶格
        for index_tuple_iteration, site_iteration in np.ndenumerate(model.dimension_array[..., 0]):
            index_tuple_list_temp_temp = []
            flag = True
            for i in range(len(inner_index_list)):
                index_tuple_temp = ()
                for j in range(len(index_tuple_iteration)):
                    index_tuple_temp = index_tuple_temp + (index_tuple_list_temp_temp[j] + cell_vector_list[j],)
                index_tuple_temp = index_tuple_temp + (inner_index_list[i],)
                index_tuple_list_temp_temp.append(index_tuple_temp)

            ##  应用周期条件
            bc=model.periodicity
            index_tuple_list_temp=[]
            for j in range(len(index_tuple_list_temp_temp)):
                index_tuple_temp=()
                index_tuple_temp_temp=index_tuple_list_temp_temp[j]
                if isinstance(bc,bool):
                    if bc:
                        for i in range(len(index_tuple_temp_temp)-1):
                            index_tuple_temp=index_tuple_temp+(np.mod(index_tuple_temp_temp[i],model.cell_period_list[i]),)
                        index_tuple_temp = index_tuple_temp + (index_tuple_temp_temp[-1],)
                elif isinstance(bc,list):
                    for i in range(len(index_tuple_temp_temp) - 1):
                        if bc[i]:
                            index_tuple_temp = index_tuple_temp + (np.mod(index_tuple_temp_temp[i], model.cell_period_list[i]),)
                    index_tuple_temp = index_tuple_temp + (index_tuple_temp_temp[-1],)
                index_tuple_list_temp.append(index_tuple_temp)

            ##  添加判断格点是否跳出范围
            for i in range(len(index_tuple_list_temp)):
                if not all(index_tuple_list_temp[i][it] < model.cell_period_list[it] for it in range(len(index_tuple_list_temp[i]))):
                    flag = False
                    break

            ##  格点在晶格内时计算算符乘积
            if flag:
                for i in range(len(op_list)):
                    if i == 0:
                        qobj = get_local_operator_qutip(model,index_tuple_list_temp[i], op_list[i])
                    else:
                        qobj = qobj * get_local_operator_qutip(model,index_tuple_list_temp[i], op_list[i])

    ##  SECTION：根据含时返回结果-------------------------------------------------------------------
    if term.time:
        function_temp=FunctionQutip(term.function,term.function_params)
        return [qobj, function_temp.MyFunction]
    else:
        if term.effect=='noise':
            return [qobj,term.strength]
        elif term.effect=='hamiltonian' or term.effect=='lindblad' or term.effect=='observe':
            return qobj*term.strength
        else:
            raise TypeError('不支持的作用量类型')
