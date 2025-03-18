import numpy as np
from tenpy import MPS
from tenpy.algorithms import dmrg
from Physics.QuantumSimulation.State.StateTenpy.StateTenpy import StateTenpy
from Physics.QuantumSimulation.Format.ModelFormat.ModelFormat import ModelFormat
from Physics.QuantumSimulation.Algorithm.Interface.InterfaceTenpy.GetModelTenpy import get_model_tenpy
from Physics.QuantumSimulation.Algorithm.Interface.InterfaceTenpy.GetLatticeTenpy import get_lattice_tenpy


#%%  KEY：基于tenpy的基态求解
"""
ground_state_solver_qutip：计算基态能量和基态态矢
input.model_format：ModelFormat对象，算符所在的模型
output.groundenergy：float对象，基态能量
output.groundstate：State对象，基态态矢
influence：本函数不改变参数对象
"""
def ground_state_solver_tenpy(model_format)->tuple[float, StateTenpy]:
    ##  SECTION：标准化-----------------------------------------------------------------------------
    assert isinstance(model_format,ModelFormat), '参数model_origin必须是ModelFormat对象'
    site_number=np.prod(model_format.cell_period_list)*len(model_format.site_dimension_list)

    ##  SECTION：计算-------------------------------------------------------------------------------
    dmrg_params = {
        'mixer': None,
        'max_E_err': 1.e-10,
        'trunc_params': {
            'chi_max': site_number * 5,
            'svd_min': 1.e-10
        },
        'combine': True
    }
    lattice_tenpy=get_lattice_tenpy(model_format)
    psi=MPS.from_product_state(lattice_tenpy.mps_sites(), [0] * site_number)
    model_tenpy, C_list, N_list = get_model_tenpy(model_format)
    info = dmrg.run(psi, model_tenpy, dmrg_params)
    psi_final=StateTenpy(psi,model_format.dimension_array)

    ##  SECTION：返回结果---------------------------------------------------------------------------
    return np.real(info['E']), psi_final
