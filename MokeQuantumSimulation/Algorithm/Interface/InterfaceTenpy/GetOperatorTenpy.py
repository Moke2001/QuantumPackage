from MokeQuantumSimulation.Algorithm.Interface.InterfaceTenpy.ModelCreator import ModelCreator
from MokeQuantumSimulation.Format.ModelFormat.ModelFormat import ModelFormat
from MokeQuantumSimulation.Format.TermFormat.TermFormat import TermFormat
from MokeQuantumSimulation.Format.TermFormat.TermsFormat import TermsFormat
from MokeQuantumSimulation.Algorithm.Interface.InterfaceTenpy.ModelPure import ModelPure


#%%  KEY：返回tenpy算符对象
"""
input.model：ModelFormat对象，模型
input.term：TermsFormat对象或TermFormat对象，格式算符
output：MPO对象，tenpy算符
influence：本函数不改变参数对象
"""
def get_operator_tenpy(model_format,term):
    ##  SECTION：标准化-----------------------------------------------------------------------------
    assert isinstance(model_format,ModelFormat),'参数model_format必须是ModelFormat对象'
    assert isinstance(term,TermFormat) or isinstance(term,TermsFormat),'参数term必须是TermsFormat对象或TermFormat对象'

    ##  SECTION：构造MPO----------------------------------------------------------------------------
    ##  利用模型构造MPO
    model_temp = model_format.copy()
    model_temp.clear()
    model_temp.push(term)

    ##  基于tenpy的计算
    model_params = {
        'model_format': model_temp,
        'time'     : 0,
    }

    ##  含时的情况
    if term.time:
        model_temp=ModelPure(model_params)
        return [model_temp.calc_H_MPO(),term.function,term.function_params]

    ##  不含时的情况
    else:
        if term.effect=='hamiltonian' or term.effect=='lindblad' or term.effect=='observe':
            model_temp=ModelCreator(model_params)
            return model_temp.calc_H_MPO()
        elif term.effect=='noise':
            model_temp=ModelPure(model_params)
            return [model_temp.calc_H_MPO(),term.strength]
        else:
            raise TypeError('参数类型错误')


