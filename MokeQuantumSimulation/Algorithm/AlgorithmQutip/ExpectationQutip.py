from qutip import expect
from MokeQuantumSimulation.Format.ModelFormat.ModelFormat import ModelFormat
from MokeQuantumSimulation.State.StateNumpy.StateNumpy import StateNumpy
from MokeQuantumSimulation.State.StatePreparer.StatePreparer import state_preparer
from MokeQuantumSimulation.Format.TermFormat.TermFormat import TermFormat
from MokeQuantumSimulation.Format.TermFormat.TermsFormat import TermsFormat
from MokeQuantumSimulation.Algorithm.Interface.InterfaceQutip.GetOperatorQutip import get_operator_qutip
from MokeQuantumSimulation.State.StateQutip.StateQutip import StateQutip
from MokeQuantumSimulation.State.StateTenpy.StateTenpy import StateTenpy


#%%  USER：基于qutip计算态矢上可观测量的期望值
"""
input.model_format：ModelFormat对象，模型
input.state_origin：state对象，初始态矢
input.term：TermsFormat对象或TermFormat对象，可观测量算符
output：float对象，期望值
influence：本函数不改变参数对象
"""
def expectation_qutip(model_format,state_origin,term):
    ##  SECTION：标准化-----------------------------------------------------------------------------
    assert isinstance(model_format, ModelFormat), '参数model_format必须是ModelFormat对象'
    assert isinstance(state_origin, StateQutip) or isinstance(state_origin, StateNumpy) or isinstance(state_origin, StateTenpy), '参数state_format必须是State对象'
    assert isinstance(term, TermsFormat) or isinstance(term, TermFormat), '参数term必须是TermsFormat对象或TermFormat对象'
    state=state_preparer(state_origin,'qutip')

    ##  SECTION：基于qutip计算----------------------------------------------------------------------
    qobj=0
    if isinstance(term, TermFormat):
        assert not term.time, '算符不允许含时'
        qobj = get_operator_qutip(model_format,term)
    elif isinstance(term, TermsFormat):
        for j in range(len(term.get_terms())):
            assert not term.get_term(j).time,'算符不允许含时'
            qobj=qobj+get_operator_qutip(model_format, term.get_term(j))

    ##  SECTION：返回结果---------------------------------------------------------------------------
    return expect(qobj, state.vector)