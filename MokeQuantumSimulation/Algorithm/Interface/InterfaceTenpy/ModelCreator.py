from tenpy import Lattice
from tenpy.models import CouplingMPOModel
from MokeQuantumSimulation.Algorithm.Interface.InterfaceTenpy.GetLatticeTenpy import get_lattice_tenpy
from MokeQuantumSimulation.Format.TermFormat.CouplingTermFormat import CouplingTermFormat
from MokeQuantumSimulation.Format.TermFormat.MultiTermFormat import MultiTermFormat
from MokeQuantumSimulation.Format.TermFormat.OnsiteTermFormat import OnsiteTermFormat
from MokeQuantumSimulation.Format.TermFormat.OverallCouplingTermFormat import OverallCouplingTermFormat
from MokeQuantumSimulation.Format.TermFormat.OverallMultiTermFormat import OverallMultiTermFormat
from MokeQuantumSimulation.Format.TermFormat.OverallOnsiteTermFormat import OverallOnsiteTermFormat
from MokeQuantumSimulation.Format.TermFormat.TermsFormat import TermsFormat


class ModelCreator(CouplingMPOModel):
    #%%  KEY：构造模型的作用量
    def init_terms(self, model_params):
        ##  SECTION：标准化-------------------------------------------------------------------------
        term_list = model_params.get('model_format', None)
        lattice = get_lattice_tenpy(model_params.get('model_format', None))
        time = model_params.get('time', 0)
        assert isinstance(term_list, TermsFormat), 'term_list must be of type TermList'
        assert isinstance(lattice, Lattice), 'lattice must be of type Lattice'

        ##  SECTION：执行作用量添加-----------------------------------------------------------------
        for i in range(term_list.number):
            term_temp = term_list.get_term(i)  # 获取当前作用量
            if term_temp.effect == 'hamiltonian' or term_temp.effect == 'observe':

                ##  SECTION：单局域作用量
                if isinstance(term_temp, OnsiteTermFormat):
                    mps_index = lattice.lat2mps_idx(term_temp.get_position())
                    if term_temp.time:
                        strength = term_temp.function(time, term_temp.function_params)
                    else:
                        strength = term_temp.strength
                    self.add_onsite_term(strength, mps_index, term_temp.get_op())

                ##  SECTION：单两体相互作用量
                elif isinstance(term_temp, CouplingTermFormat):
                    position_0, position_1 = term_temp.get_position()
                    mps_index_0 = lattice.lat2mps_idx(position_0)
                    mps_index_1 = lattice.lat2mps_idx(position_1)
                    op_0, op_1 = term_temp.get_op()
                    if term_temp.time:
                        strength = term_temp.function(time, term_temp.function_params)
                    else:
                        strength = term_temp.strength
                    self.add_coupling_term(strength, mps_index_0, mps_index_1, op_0, op_1)

                ##  SECTION：单多体相互作用量
                elif isinstance(term_temp, MultiTermFormat):
                    op_list = term_temp.get_op()
                    position_list = term_temp.get_position()
                    mps_index_list = [lattice.lat2mps_idx(position) ** 2 for position in position_list]
                    if term_temp.time:
                        strength = term_temp.function(time, term_temp.function_params)
                    else:
                        strength = term_temp.strength
                    self.add_multi_coupling_term(strength,mps_index_list,op_list,['Id']*(len(position_list)-1))

                ##  SECTION：遍历局域相互作用量
                elif isinstance(term_temp, OverallOnsiteTermFormat):
                    cell_index = term_temp.get_position()
                    op = term_temp.get_op()
                    if term_temp.time:
                        strength = term_temp.function(time, term_temp.function_params)
                    else:
                        strength = term_temp.strength
                    self.add_onsite(strength, cell_index, op, category=None, plus_hc=False)

                ##  SECTION：遍历两体相互作用量
                elif isinstance(term_temp, OverallCouplingTermFormat):
                    cell_index_0, cell_index_1, vector = term_temp.get_position()
                    op_0, op_1 = term_temp.get_op()
                    if term_temp.time:
                        strength = term_temp.function(time, term_temp.function_params)
                    else:
                        strength = term_temp.strength
                    self.add_coupling(strength, cell_index_0, op_0, cell_index_1, op_1, vector)

                ##  SECTION：遍历多体相互作用量
                elif isinstance(term_temp, OverallMultiTermFormat):
                    cell_list, vector_list = term_temp.get_position()
                    op_list = term_temp.get_op()
                    if term_temp.time:
                        strength = term_temp.function(time, term_temp.function_params)
                    else:
                        strength = term_temp.strength
                    unit_list = []
                    for i in range(len(cell_list)):
                        unit_list.append((op_list[i], vector_list[i], cell_list[i]))
                    self.add_multi_coupling(strength, unit_list, op_list)

                ##  SECTION：抛出类型错误
                else:
                    raise TypeError('Unexpected term type')


    # %%  KEY：从模型参数中获取晶格对象
    def init_lattice(self, model_params):
        return get_lattice_tenpy(model_params.get('model_format', None))
