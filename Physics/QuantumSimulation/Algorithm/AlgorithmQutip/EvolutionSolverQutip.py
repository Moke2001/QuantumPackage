import numpy as np
from Physics.QuantumSimulation.State.StatePreparer.StatePreparer import state_preparer
from Physics.QuantumSimulation.State.StateQutip.StateQutip import StateQutip
from Physics.QuantumSimulation.Format.ModelFormat.ModelFormat import ModelFormat
from Physics.QuantumSimulation.Format.TermFormat.TermFormat import TermFormat
from Physics.QuantumSimulation.Format.TermFormat.TermsFormat import TermsFormat
from qutip import *
from Physics.QuantumSimulation.Algorithm.Interface.InterfaceQutip.GetModelQutip import get_model_qutip
from Physics.QuantumSimulation.Algorithm.Interface.InterfaceQutip.GetOperatorQutip import get_operator_qutip


#%%  USER：基于qutip计算态矢演化过程中期望变化和演化后的态矢
"""
evolution_solver_qutip：基于qutip计算态矢演化过程中期望变化和演化后的态矢
input.model_format：ModelFormat对象，算符所在的模型
input.psi_origin：State对象，初始态矢
input.expectation_terms：list of TermFormat对象，待求期望的可观测量
input.t_list：np.ndarray对象，时间序列
output.psi_final：State对象，末态态矢
output.data_list：list of np.ndarray对象，各个可观测量期望随时间变化，与t_list对应
influence：本函数不改变参数对象
"""
def evolution_solver_qutip(model_format:ModelFormat, psi_origin, expectation_terms:list, t_list:np.ndarray):
    ##  SECTION：标准化-----------------------------------------------------------------------------
    assert isinstance(t_list,np.ndarray),'参数t_list必须是np.ndarray对象'
    assert isinstance(model_format,ModelFormat), '参数model_origin必须是ModelFormat对象'
    assert isinstance(expectation_terms,list),'参数expectation_terms必须是list对象'
    assert all(it.time==False for it in expectation_terms),'参数expectation_terms中的作用量不允许含时'
    psi=state_preparer(psi_origin,'qutip')

    ##  SECTION：求可观测量的qutip形式算符----------------------------------------------------------
    ##  计算期望作用量的qutip算符形式
    E_list=[]
    for i in range(len(expectation_terms)):
        if isinstance(expectation_terms[i],TermFormat):
            E_list.append(get_operator_qutip(model_format,expectation_terms[i]) * expectation_terms[i].get_strength())
        elif isinstance(expectation_terms[i],TermsFormat):
            qobj=0
            for j in range(len(expectation_terms[i].get_terms())):
                qobj= qobj + get_operator_qutip(model_format,expectation_terms[i].get_temrs()[j]) * expectation_terms[i].get_terms()[j].get_strength()
            E_list.append(qobj)

    ##  用qutip计算
    H_list, C_list, N_list = get_model_qutip(model_format)
    result_list = mesolve(H_list, rho0=psi.vector, tlist=t_list, c_ops=C_list, e_ops=E_list, args={}).expect
    psi_result = mesolve(H_list, rho0=psi.vector, tlist=t_list, c_ops=C_list, args={}).states[-1]

    ##  期望序列赋值
    data_list=[]
    for i in range(len(result_list)):
        data_list.append(result_list[i])

    ##  态矢赋值
    state_result=StateQutip(psi_result,model_format.dimension_array)

    ##  SECTION：返回结果---------------------------------------------------------------------------
    return state_result,data_list
