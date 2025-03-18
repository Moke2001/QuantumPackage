import numpy as np
from qutip import Qobj, identity, tensor
from Physics.QuantumSimulation.Format.ModelFormat.ModelFormat import ModelFormat


#%%  KEY：将一个局域矩阵转化为Qobj对象
"""
input.model_format：ModelFormat对象，模型
input.index_tuple：tuple对象，格位点
input.matrix：np.ndarray对象，局域算符
output：Qobj对象，qutip算符
influence：本函数不改变参数对象
"""
def get_local_operator_qutip(model_format,index_tuple, name)->Qobj:
    ##  SECTION：标准化-----------------------------------------------------------------------------
    assert isinstance(model_format,ModelFormat),'参数model_format必须是ModelFormat对象'
    assert isinstance(index_tuple,tuple),'参数index_tuple必须是tuple对象'
    assert isinstance(name,str),'参数name必须是str对象'

    ##  SECTION：计算-------------------------------------------------------------------------------
    qobj = None
    for index_tuple_temp, dimension_temp in np.ndenumerate(model_format.dimension_array):
        ##  第一个位点
        if qobj is None:
            if index_tuple_temp==index_tuple:
                qobj=Qobj(model_format.local_operator_dictionary[name])
            else:
                qobj=identity(dimension_temp)

        ##  其他位点
        else:
            if index_tuple_temp==index_tuple:
                qobj=tensor(qobj,Qobj(model_format.local_operator_dictionary[name]))
            else:
                qobj = tensor(qobj, identity(dimension_temp))

    ##  SECTION：返回结果---------------------------------------------------------------------------
    return qobj
