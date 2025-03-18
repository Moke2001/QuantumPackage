import numpy as np

from Physics.QuantumComputation.CliffordSimulator.FermionicComputer import FermionicComputer
from Physics.QuantumComputation.Code.QuantumCode.Fermion.FermionicCode import FermionicCode
from Physics.QuantumComputation.Code.QuantumCode.Fermion.FermionicColorCode import FermionicColorCode
from Physics.QuantumComputation.Code.QuantumCode.Fermion.MajoranaOperator import MajoranaOperator
from Physics.QuantumComputation.Code.QuantumCode.Fermion.ZechuanLatticeSurgery import ZechuanLatticeSurgery


def ErrorType():
    code_memory = FermionicCode()
    code_memory.define_qubit(10)
    code_memory.push(MajoranaOperator([0,1,5,6],[],1))
    code_memory.push(MajoranaOperator([0,3,5,8],[],1))
    code_memory.push(MajoranaOperator([2,4,7,9],[],1))
    code_memory.push(MajoranaOperator([],[0,1,5,6],1))
    code_memory.push(MajoranaOperator([],[0,3,5,8],1))
    code_memory.push(MajoranaOperator([],[2,4,7,9],1))
    code_memory.logical_operator_list_x=[MajoranaOperator([0,1,2,3,4],[],1),MajoranaOperator([5,6,7,8,9],[],1)]
    code_memory.logical_operator_list_z=[MajoranaOperator([],[0,1,2,3,4],1),MajoranaOperator([],[5,6,7,8,9],1)]
    code=ZechuanLatticeSurgery(code_memory,0)
    code_processor=FermionicColorCode(5)
    code_processor.index_map(code.number_qubit,np.arange(10,10+code_processor.number_qubit))
    code_memory.index_map(code.number_qubit,np.arange(10))


    computer=FermionicComputer()
    computer.define(code.number_qubit,0)
    stabilizers=[]
    for i in range(len(code_memory.check_list)):
        stabilizers.append(code_memory.check_list[i])
    for i in range(len(code_processor.check_list)):
        stabilizers.append(code_processor.check_list[i])
    stabilizers.append(MajoranaOperator([0,1,2,3,4],[0,1,2,3,4],1))
    stabilizers.append(MajoranaOperator([5,6,7,8,9],[5,6,7,8,9],1))


    for i in range(code.number_qubit-10-19):
        n=i+29
        stabilizers.append(MajoranaOperator([n],[n],1))
    x=code_processor.logical_operator_list_x[0].x_vector
    stabilizers.append(MajoranaOperator(x.copy(),x.copy(),1))
    computer.initialize(stabilizers)
    error=MajoranaOperator([20,21,37,38],[],1)
    for i in range(len(error.x_vector)):
        computer.pip_x(error.x_vector[i])
    computer.correct(code,'common')
    for i in range(code.number_qubit-10-19):
        n=i+29
        computer.measure(MajoranaOperator([n],[n],1))
    for i in range(len(code_memory.check_list)):
        value,flag=computer.get_value(code_memory.check_list[i])
        if value==-1:
            code_memory.check_list[i].coff=-code_memory.check_list[i].coff
    for i in range(len(code_processor.check_list)):
        value, flag = computer.get_value(code_processor.check_list[i])
        if value == -1:
            code_processor.check_list[i].coff = -code_processor.check_list[i].coff


    pass


if __name__ == '__main__':
    for i in range(20):
        ErrorType()