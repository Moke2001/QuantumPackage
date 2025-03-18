import numpy as np
from matplotlib import pyplot as plt
from Physics.QuantumSimulation.Algorithm.AlgorithmSolver.AlgorithmSolver import AlgorithmSolver
from Physics.QuantumSimulation.Format.TermFormat.OverallCouplingTermFormat import OverallCouplingTermFormat
from Physics.QuantumSimulation.State.StateNumpy.StateNumpy import StateNumpy
from Physics.QuantumSimulation.Format.ModelFormat.ModelFormat import ModelFormat
from Physics.QuantumSimulation.Format.TermFormat.OnsiteTermFormat import OnsiteTermFormat
from Physics.QuantumSimulation.Format.TermFormat.OverallOnsiteTermFormat import OverallOnsiteTermFormat


#%%  USER：定义时变函数
def zeeman(t,args):
    g_max=args.get('g_max')
    omega=args.get('omega')
    return np.sin(omega*t)*g_max


#%%  USER：主函数
def example():
    ##  SECTION：定义模型并更新晶格-----------------------------------------------------------------
    model=ModelFormat()
    model.set_period([5])
    model.set_cell_vector([np.array([1])])
    model.set_site_dimension([2,2])
    model.set_site_vector([np.array([0]),np.array([0.5])])
    model.set_periodicity(False)

    ##  SECTION：定义哈密顿量-----------------------------------------------------------------------
    ##  含时哈密顿量项
    function_params={
        'g_max':0.5,
        'omega':1
    }
    term = OverallOnsiteTermFormat('field', 'hamiltonian', 0, 'sigmax',zeeman,function_params)
    model.push(term)

    ##  不含时哈密顿量项
    term=OverallCouplingTermFormat('interaction','hamiltonian',0,0,(1,),'sigmaz','sigmaz',1)
    model.push(term)
    term = OverallCouplingTermFormat('interaction', 'hamiltonian', 1, 1, (1,), 'sigmaz', 'sigmaz', 1)
    model.push(term)

    ##  SECTION：定义初态---------------------------------------------------------------------------
    x=(1, 0)
    y=(0, 1)
    state_array = np.array([[y,y], [y,y], [y,x], [x,y], [y,x]])
    state=StateNumpy.FromLocalState(state_array)

    ##  SECTION：定义可观测量-----------------------------------------------------------------------
    expectation_terms = []
    for i in range(5):
        term=OnsiteTermFormat('observe','observe',(i,0),'sigmaz',1)
        expectation_terms.append(term)

    ##  SECTION：定义模拟---------------------------------------------------------------------------
    t_list=np.linspace(0,1,20)
    psi_result,data_list=AlgorithmSolver.EvolutionSolver(model,state,expectation_terms,t_list,'tenpy')

    ##  SECTION：输出结果---------------------------------------------------------------------------
    plt.plot(t_list,data_list[1])
    plt.show()


if __name__=="__main__":
    example()
