from Physics.QuantumSimulation.State.StateQutip.StateQutip import StateQutip
from Physics.QuantumSimulation.Format.ModelFormat.ModelFormat import ModelFormat
from Physics.QuantumSimulation.Algorithm.Interface.InterfaceQutip.GetModelQutip import get_model_qutip


#%%  USER：基于qutip计算模型基态和基态能量
"""
input.model_format：ModelFormat对象，算符所在的模型
output.groundenergy：float对象，基态能量
output.groundstate：State对象，基态态矢
influence：本函数不改变参数对象
"""
def ground_state_solver_qutip(model_origin:ModelFormat)->tuple[float,StateQutip]:
    ##  SECTION：标准化-----------------------------------------------------------------------------
    assert isinstance(model_origin,ModelFormat), '参数model_origin必须是ModelFormat对象'

    ##  SECTION：计算-------------------------------------------------------------------------------
    H_list, C_list, N_list, function_params = get_model_qutip(model_origin)
    groundenergy, temp=H_list[0].groundstate()
    groundstate=StateQutip(temp,model_origin.dimension_array)

    ##  SECTION：返回结果---------------------------------------------------------------------------
    return groundenergy, groundstate