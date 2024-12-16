from Framework.Format.ModelFormat.ModelFormat import ModelFormat
from Framework.ModelPreparer.ModelPreparer import model_preparer


"""
energy_spectrum_solver_tenpy：计算本征值和本征矢
input.model_origin：ModelFormat对象，算符所在的模型
output.eigenvalues：list of float对象，从小到大排列的本征值
output.eigenvectors：list of Qobj对象，与本征值列表一一对应的本征矢量
"""
def energy_spectrum_solver_tenpy(model_origin):
    ##  标准化
    assert isinstance(model_origin, ModelFormat), '参数model_origin必须是ModelFormat对象'
    model = model_preparer(model_origin, 'qutip')

    ##  TODO：返回结果
    raise NotImplemented