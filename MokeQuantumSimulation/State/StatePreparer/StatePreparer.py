import numpy as np
from MokeQuantumSimulation.State.StateNumpy.StateNumpy import StateNumpy
from MokeQuantumSimulation.State.StateQutip.StateQutip import StateQutip
from MokeQuantumSimulation.State.StateTenpy.StateTenpy import StateTenpy


#%%  KEY：将态矢准备为对应的形式
"""
input.state：State对象，态矢
input.type：str对象，目标态矢形式
input.args：ModelFormat对象，如果目标是mps要用到模型
output：State对象，转化后的态矢
influence：本函数不改变参数对象
"""
def state_preparer(state,type):
    ##  SECTION：标准化
    assert isinstance(state,StateNumpy) or isinstance(state,StateTenpy) or isinstance(state,StateQutip),'参数state类型有误'
    assert isinstance(type,str),'参数type必须是str对象'

    ##  SECTION：目标是qutip态矢--------------------------------------------------------------------
    if type=='qutip':
        if isinstance(state,StateNumpy):
            return StateQutip.FromStateNumpy(state)
        elif isinstance(state,StateTenpy):
            raise NotImplemented
        elif isinstance(state,StateQutip):
            return state
        else:
            raise TypeError

    ##  SECTION：目标是tenpy态矢--------------------------------------------------------------------
    elif type=='tenpy':
        if isinstance(state,StateNumpy):
            return StateTenpy.FromStateNumpy(state)
        elif isinstance(state,StateTenpy):
            return state
        elif isinstance(state,StateQutip):
            raise NotImplemented
        else:
            raise TypeError

    ##  SECTION：目标是numpy态矢--------------------------------------------------------------------
    elif type=='numpy':
        if isinstance(state,StateNumpy):
            return state
        elif isinstance(state,StateTenpy):
            raise NotImplemented
        elif isinstance(state,StateQutip):
            shape=()
            for index_tuple, dimension_temp in np.ndenumerate(state.dimension_array):
                shape=shape+(dimension_temp,)
            return StateNumpy(state.vector.full().reshape(shape),state.dimension_array)
        else:
            raise TypeError

    ##  SECTION：不支持的类型-----------------------------------------------------------------------
    else:
        raise TypeError