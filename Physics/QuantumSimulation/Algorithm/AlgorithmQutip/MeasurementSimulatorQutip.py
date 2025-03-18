import random
import numpy as np
from Physics.QuantumSimulation.Algorithm.AlgorithmQutip.ApplyQutip import apply_qutip
from Physics.QuantumSimulation.Algorithm.AlgorithmQutip.ExpectationQutip import expectation_qutip
from Physics.QuantumSimulation.State.StatePreparer.StatePreparer import state_preparer
from Physics.QuantumSimulation.State.StateQutip.StateQutip import StateQutip
from Physics.QuantumSimulation.Format.ModelFormat.ModelFormat import ModelFormat


#%%  USER：计算态矢测量结果与坍缩态
"""
input.model_format：ModelFormat对象，算符所在的模型
input.psi_origin：State对象，初始态矢
input.projector_list：list of TermFormat对象，投影算符
input.eigenvalue_list：list of float对象，投影算符对应测量值
output.psi_final：State对象，测量后坍缩到的态矢
output.value：float对象，测量值
influence：本函数不改变参数对象
"""
def measurement_simulator_qutip(model_format, psi_origin, projector_list, eigenvalue_list)->tuple[float,StateQutip]:
    ##  SECTION：标准化-----------------------------------------------------------------------------
    assert isinstance(model_format, ModelFormat), '参数model_origin必须是ModelFormat对象'
    assert isinstance(projector_list, list),'参数projector_list必须是list对象'
    assert isinstance(eigenvalue_list, list),'参数eigenvalue_list必须是list对象'
    psi=state_preparer(psi_origin,'qutip')

    ##  SECTION：随机坍缩模拟-----------------------------------------------------------------------
    P_list = []
    for i in range(len(projector_list)):
        P_list.append(expectation_qutip(model_format,psi,projector_list[i]))
    assert np.abs(1-np.sum(P_list))<0.0001,'投影算符不归一化'
    index = random.choices(range(len(P_list)), weights=P_list, k=1)[0]
    psi_final=apply_qutip(model_format,psi,projector_list[index],True)
    value=eigenvalue_list[index]

    ##  SECTION：返回结果---------------------------------------------------------------------------
    return value,psi_final
