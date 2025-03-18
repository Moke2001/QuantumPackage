from Physics.QuantumComputation.Code.ClassicalCode.ProjectiveCode import ProjectiveCode
from Physics.QuantumComputation.Code.QuantumCode.Fermion.FermionicCode import FermionicCode
from Physics.QuantumComputation.Code.QuantumCode.Fermion.FermionicColorCode import FermionicColorCode
from Physics.QuantumComputation.Code.QuantumCode.Fermion.FermionicLatticeSurgery import FermionicLatticeSurgery
from Physics.QuantumComputation.Code.QuantumCode.Fermion.MajoranaOperator import MajoranaOperator


def EulerDecompositionTest():
    code_memory = FermionicCode()
    code_memory.linear_combine(ProjectiveCode(3))

    opx0 = MajoranaOperator([0, 7, 25, 41, 51], [], 1)
    opx1 = MajoranaOperator([1, 11, 47, 50, 62], [], 1)
    opx2 = MajoranaOperator([2, 10, 27, 43, 54], [], 1)
    opx3 = MajoranaOperator([3, 13, 22, 31, 58], [], 1)
    opz0 = MajoranaOperator([], [0, 7, 25, 41, 51], 1)
    opz1 = MajoranaOperator([], [1, 11, 47, 50, 62], 1)
    opz2 = MajoranaOperator([], [2, 10, 27, 43, 54], 1)
    opz3 = MajoranaOperator([], [3, 13, 22, 31, 58], 1)
    code_memory.logical_operator_list_x = [opx0, opx1, opx2, opx3]
    code_memory.logical_operator_list_z = [opz0, opz1, opz2, opz3]
    code_processor=FermionicColorCode(5)
    code=FermionicLatticeSurgery(code_memory, code_processor,1,0)


if __name__ == '__main__':
    EulerDecompositionTest()