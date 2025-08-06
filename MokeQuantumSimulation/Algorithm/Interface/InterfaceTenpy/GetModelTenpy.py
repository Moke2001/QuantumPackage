from tenpy.models import CouplingMPOModel
from MokeQuantumSimulation.Algorithm.Interface.InterfaceTenpy.GetOperatorTenpy import get_operator_tenpy
from MokeQuantumSimulation.Algorithm.Interface.InterfaceTenpy.ModelCreator import ModelCreator
from MokeQuantumSimulation.Format.ModelFormat.ModelFormat import ModelFormat


#%%  KEY：根据格式模型返回用于tenpy计算的模型
"""
input.model_format：ModelFormat对象，模型
output.model_tenpy：Model对象，tenpy模型
output.C_list：list of MPO对象，tenpy中Lindblad算符
output.N_list：list of MPO对象，tenpy中噪声算符
influence：本函数不改变参数对象
"""
def get_model_tenpy(model_format)->tuple[CouplingMPOModel,list,list]:
    ##  SECTION：标准化-----------------------------------------------------------------------------
    assert isinstance(model_format,ModelFormat),'参数model_format必须是ModelFormat对象'

    ##  SECTION：构造模型---------------------------------------------------------------------------
    model_params = {
        'model_format': model_format,
        'time'     : 0,
    }
    model_tenpy = ModelCreator(model_params)

    ##  SECTION：构造Lindblad算符列表和噪声算符列表-------------------------------------------------
    C_list=[]
    N_list=[]
    for term in model_format.get_terms():
        if term.effect=='noise':
            N_list.append(get_operator_tenpy(model_format, term))
        elif term.effect=='lindblad':
            C_list.append(get_operator_tenpy(model_format, term))
        elif term.effect=='hamiltonian':
            pass
        else:
            raise TypeError('不支持的作用量类型')

    ##  SECTION：返回结果---------------------------------------------------------------------------
    return model_tenpy, C_list, N_list


