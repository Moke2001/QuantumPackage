from MokeQuantumSimulation.Format.ModelFormat.ModelFormat import ModelFormat
from MokeQuantumSimulation.State.StateNumpy.StateNumpy import StateNumpy
from MokeQuantumSimulation.State.StatePreparer.StatePreparer import state_preparer
from MokeQuantumSimulation.State.StateQutip.StateQutip import StateQutip
from MokeQuantumSimulation.Format.TermFormat.TermFormat import TermFormat
from MokeQuantumSimulation.Format.TermFormat.TermsFormat import TermsFormat
from MokeQuantumSimulation.Algorithm.Interface.InterfaceQutip.GetOperatorQutip import get_operator_qutip
from MokeQuantumSimulation.State.StateTenpy.StateTenpy import StateTenpy


#%%  USER：基于qutip计算算符作用在态矢上后的态矢
"""
input.model_format：ModelFormat对象，模型
input.state_origin：state对象，初始态矢
input.term：TermsFormat对象或TermFormat对象，作用的算符
output：StateQutip对象，qutip形式态矢
influence：本函数不改变参数对象
"""
def apply_qutip(model_format, state_origin, term,normalize)->StateQutip:
    ##  SECTION：标准化-----------------------------------------------------------------------------
    assert isinstance(model_format,ModelFormat),'参数model_format必须是ModelFormat对象'
    assert isinstance(state_origin,StateQutip) or isinstance(state_origin,StateNumpy) or isinstance(state_origin,StateTenpy), '参数state_format必须是State对象'
    assert isinstance(term,TermsFormat) or isinstance(term,TermFormat),'参数term必须是TermsFormat对象或TermFormat对象'
    state=state_preparer(state_origin, 'qutip')

    ##  SECTION：基于qutip计算作用结果--------------------------------------------------------------
    ##  单一作用量
    if isinstance(term, TermFormat):
        assert not term.time,'算符不允许含时'
        qobj = get_operator_qutip(model_format,term)*term.get_strength()
        result=StateQutip(qobj*state.vector,model_format.dimension_array)
        if normalize:
            result.normalize()
        return result

    ##  作用量列表
    elif isinstance(term, TermsFormat):
        qobj=0
        for j in range(len(term.get_terms())):
            assert not term.get_term(j).time,'算符不允许含时'
            qobj=qobj+get_operator_qutip(model_format,term.get_term(j))
        result = StateQutip(qobj * state.vector, model_format.dimension_array)
        if normalize:
            result.normalize()