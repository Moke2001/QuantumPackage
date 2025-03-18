from Physics.QuantumSimulation.State.StatePreparer.StatePreparer import state_preparer
from Physics.QuantumSimulation.State.StateTenpy.StateTenpy import StateTenpy
from Physics.QuantumSimulation.Format.ModelFormat.ModelFormat import ModelFormat
from Physics.QuantumSimulation.Format.TermFormat.TermFormat import TermFormat
from Physics.QuantumSimulation.Format.TermFormat.TermsFormat import TermsFormat
from Physics.QuantumSimulation.Algorithm.Interface.InterfaceTenpy.GetOperatorTenpy import get_operator_tenpy


def apply_tenpy(model_format,state,term,normalize):
    ##  SECTION：标准化-----------------------------------------------------------------------------
    assert isinstance(term, TermFormat) or isinstance(term, TermsFormat), '参数term必须是TermFormat或TermsFormat对象'
    assert isinstance(model_format, ModelFormat), '参数model_origin必须是ModelFormat对象'
    psi=state_preparer(state,'tenpy')

    ##  SECTION：基于tenpy计算作用结果--------------------------------------------------------------
    ##  单一作用量
    if isinstance(term, TermFormat):
        assert not term.time,'算符不允许含时'
        mps = get_operator_tenpy(model_format,term).apply(psi.mps.copy())
        result=StateTenpy(mps,model_format.dimension_array)
        if normalize:
            result.normalize()

    ##  作用量列表
    elif isinstance(term, TermsFormat):
        mps=None
        for j in range(len(term.get_terms())):
            assert not term.get_term(j).time,'算符不允许含时'
            if j==0:
                mps = get_operator_tenpy(model_format, term).apply(psi.mps.copy())
            else:
                mps = get_operator_tenpy(model_format, term).apply(mps)
        result = StateTenpy(mps, model_format.dimension_array)
        if normalize:
            result.normalize()
        return result