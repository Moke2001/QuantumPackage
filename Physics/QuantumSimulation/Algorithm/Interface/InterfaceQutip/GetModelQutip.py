from Physics.QuantumSimulation.Format.ModelFormat.ModelFormat import ModelFormat
from Physics.QuantumSimulation.Algorithm.Interface.InterfaceQutip.GetOperatorQutip import get_operator_qutip


#%%  KEY：得到qutip计算所用的对象
"""
input.model：ModelFormat对象，模型
output.H_list：哈密顿量列表
output.C_list：Lindblad算符列表
output.N_list：噪声算符列表
influence：本函数不改变参数对象
"""
def get_model_qutip(model):
    ##  SECTION：标准化-----------------------------------------------------------------------------
    assert isinstance(model,ModelFormat),'参数model必须是ModelFormat对象'

    ##  SECTION：返回结果赋值-----------------------------------------------------------------------
    ##  初始化
    H_list=[]
    H=0
    C_list=[]
    N_list=[]

    ##  对每个作用量赋值
    for term in model.get_terms():
        qobj=get_operator_qutip(model,term)
        if term.effect == 'hamiltonian':
            if term.time:
                H_list.append(qobj)
            else:
                H=H+qobj
        elif term.effect == 'lindbladian':
            C_list.append(qobj)
        elif term.effect == 'noise':
            N_list.append(qobj)
    H_list=[H]+H_list

    ##  SECTION：返回结果---------------------------------------------------------------------------
    return H_list,C_list,N_list


