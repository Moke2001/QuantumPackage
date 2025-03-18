from Physics.QuantumSimulation.State.StatePreparer.StatePreparer import state_preparer
from Physics.QuantumSimulation.Format.ModelFormat.ModelFormat import ModelFormat
from Physics.QuantumSimulation.Format.TermFormat.TermFormat import TermFormat
from Physics.QuantumSimulation.Format.TermFormat.TermsFormat import TermsFormat
from Physics.QuantumSimulation.Algorithm.Interface.InterfaceTenpy.GetOperatorTenpy import get_operator_tenpy


#%%  KEY：基于tenpy的期望值计算
def expectation_tenpy(model_format,state,term):
    ##  SECTION：标准化-----------------------------------------------------------------------------
    assert isinstance(term, TermFormat) or isinstance(term, TermsFormat), '参数term必须是TermFormat或TermsFormat对象'
    assert isinstance(model_format, ModelFormat), '参数model_origin必须是ModelFormat对象'
    psi=state_preparer(state,'tenpy')

    ##  SECTION：基于tenpy计算----------------------------------------------------------------------
    return get_operator_tenpy(model_format,term).expectation_value(psi.mps)