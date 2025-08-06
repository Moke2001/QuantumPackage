from MokeQuantumSimulation.Algorithm.AlgorithmQutip.ApplyQutip import apply_qutip
from MokeQuantumSimulation.Algorithm.AlgorithmQutip.EnergySpectrumSolverQutip import energy_spectrum_solver_qutip
from MokeQuantumSimulation.Algorithm.AlgorithmQutip.EvolutionSolverQutip import evolution_solver_qutip
from MokeQuantumSimulation.Algorithm.AlgorithmQutip.ExpectationQutip import expectation_qutip
from MokeQuantumSimulation.Algorithm.AlgorithmQutip.GroundStateSolverQutip import ground_state_solver_qutip
from MokeQuantumSimulation.Algorithm.AlgorithmQutip.MeasurementSimulatorQutip import measurement_simulator_qutip
from MokeQuantumSimulation.Algorithm.AlgorithmQutip.OverlapQutip import overlap_qutip
from MokeQuantumSimulation.Algorithm.AlgorithmTenpy.EnergySpectrumSolverTenpy import energy_spectrum_solver_tenpy
from MokeQuantumSimulation.Algorithm.AlgorithmTenpy.EvolutionSolverTenpy import evolution_solver_tenpy
from MokeQuantumSimulation.Algorithm.AlgorithmTenpy.ExpectationTenpy import expectation_tenpy
from MokeQuantumSimulation.Algorithm.AlgorithmTenpy.GroundStateSolverTenpy import ground_state_solver_tenpy
from MokeQuantumSimulation.Algorithm.AlgorithmTenpy.MeasurementSimulatorTenpy import measurement_simulator_tenpy
from MokeQuantumSimulation.Algorithm.AlgorithmTenpy.OverlapTenpy import overlap_tenpy
from MokeQuantumSimulation.Format.ModelFormat.ModelFormat import ModelFormat


class AlgorithmSolver:
    #%%  USER：能谱求解
    """"
    input.model：ModelFormat对象，待求解的模型
    input.solver：str对象，求解器种类
    output.eigenvalues：list of float对象，能量列表
    output.eigenstates：list of State对象，能态列表
    influence：本函数不改变参数对象
    """""
    @staticmethod
    def EnergySpectrumSolver(model,solver):
        assert isinstance(model,ModelFormat)
        assert isinstance(solver,str)
        if solver=='qutip':
            return energy_spectrum_solver_qutip(model)
        elif solver=='tenpy':
            return energy_spectrum_solver_tenpy(model)


    #%%  USER：演化求解
    """"
    input.model：ModelFormat对象，待求解的模型
    input.psi_initial：State对象，演化初态
    input.expectation_terms：list of TermFormat or TermsFormat对象，可观测量
    input.t_list：np.ndarray对象，时间序列
    input.solver：str对象，求解器种类
    output.data：list of np.ndarray对象，可观测量期望随时间的变化
    output.psi_final：State对象，演化后的态矢
    influence：本函数不改变参数对象
    """""
    @staticmethod
    def EvolutionSolver(model,psi_initial,expectation_terms,t_list,solver):
        if solver=='qutip':
            return evolution_solver_qutip(model,psi_initial,expectation_terms,t_list)
        elif solver=='tenpy':
            return evolution_solver_tenpy(model, psi_initial, expectation_terms, t_list)
        else:
            raise NotImplemented


    #%%  USER：基态求解
    """"
    input.model：ModelFormat对象，待求解的模型
    input.solver：str对象，求解器种类
    output.eigenvalue：float对象，基态能量
    output.eigenstate：State对象，基态态矢
    influence：本函数不改变参数对象
    """""
    @staticmethod
    def GroundStateSolver(model,solver):
        if solver=='qutip':
            return ground_state_solver_qutip(model)
        elif solver=='tenpy':
            return ground_state_solver_tenpy(model)
        else:
            raise NotImplemented


    #%%  USER：测量模拟
    """"
    input.model：ModelFormat对象，待求解的模型
    input.psi_initial：State对象，演化初态
    input.projector_list：list of TermFormat or TermsFormat对象，投影算符列表
    input.eigenvalue_list：list of float对象，投影算符对应的本征值
    input.solver：str对象，求解器种类
    output.value：float对象，测量结果
    output.psi_final：State对象，测量后坍缩到的态矢
    influence：本函数不改变参数对象
    """""
    @staticmethod
    def MeasurementSimulator(model, psi_initial, projector_list, eigenvalue_list,solver):
        if solver=='qutip':
            return measurement_simulator_qutip(model,psi_initial,projector_list,eigenvalue_list)
        elif solver=='tenpy':
            return measurement_simulator_tenpy(model, psi_initial, projector_list, eigenvalue_list)
        else:
            raise NotImplemented


    #%%  USER：算符作用在态矢上
    """"
    input.model：ModelFormat对象，待求解的模型
    input.psi_initial：State对象，演化初态
    input.term：TermFormat or TermsFormat对象，要作用的算符
    input.normalize：bool对象，决定作用后是否对态矢归一化
    input.solver：str对象，求解器种类
    output.psi_final：State对象，作用后的态矢
    influence：本函数不改变参数对象
    """""
    @staticmethod
    def Apply(model_format,psi_initial,term,normalize,solver):
        if solver=='qutip':
            return apply_qutip(model_format,psi_initial,term,normalize)
        elif solver=='tenpy':
            return apply_qutip(model_format,psi_initial,term,normalize)
        else:
            raise NotImplemented



    #%%  USER：求态矢可观测量期望
    """"
    input.model：ModelFormat对象，待求解的模型
    input.psi_initial：State对象，演化初态
    input.term：TermFormat or TermsFormat对象，可观测量算符
    input.solver：str对象，求解器种类
    output.data：float对象，期望值
    influence：本函数不改变参数对象
    """""
    @staticmethod
    def Expectation(model, psi_initial, term, solver):
        if solver=='qutip':
            return expectation_qutip(model,psi_initial,term)
        elif solver=='tenpy':
            return expectation_tenpy(model,psi_initial,term)
        else:
            raise NotImplemented


    #%%  USER：求态矢可观测量期望
    """"
    input.model：ModelFormat对象，待求解的模型
    input.psi_0：State对象，左矢
    input.psi_1：State对象，右矢
    input.solver：str对象，求解器种类
    output.data：float对象，期望值
    influence：本函数不改变参数对象
    """""
    @staticmethod
    def Overlap(psi_0,psi_1,solver):
        if solver=='qutip':
            return overlap_qutip(psi_0,psi_1)
        elif solver=='tenpy':
            return overlap_tenpy(psi_0,psi_1)
        else:
            raise NotImplemented
