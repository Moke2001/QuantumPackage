from MokeQuantumSimulation.State.StateQutip.StateQutip import StateQutip
from MokeQuantumSimulation.Format.ModelFormat.ModelFormat import ModelFormat
from MokeQuantumSimulation.Algorithm.Interface.InterfaceQutip.GetModelQutip import get_model_qutip


#%%  USER：基于qutip计算能谱
"""
energy_spectrum_solver_qutip：计算本征值和本征矢
input.model_format：ModelFormat对象，算符所在的模型
output.eigenvalues：list of float对象，从小到大排列的本征值
output.eigenvectors：list of Qobj对象，与本征值列表一一对应的本征矢量
"""
def energy_spectrum_solver_qutip(model_origin: ModelFormat) -> tuple[list, list]:
    ##  SECTION：标准化-----------------------------------------------------------------------------
    assert isinstance(model_origin, ModelFormat), '参数model_origin必须是ModelFormat对象'

    ##  SECTION：计算-------------------------------------------------------------------------------
    ##  获得模型并计算
    H_list, N_list, C_list = get_model_qutip(model_origin)
    eigenvalues, temp = H_list[0].eigenstates()

    ##  打包态矢
    eigenvectors = []
    for i in range(len(temp)):
        eigenvectors.append(StateQutip(temp[i],model_origin.dimension_array))

    ##  SECTION：返回结果---------------------------------------------------------------------------
    return eigenvalues, eigenvectors
