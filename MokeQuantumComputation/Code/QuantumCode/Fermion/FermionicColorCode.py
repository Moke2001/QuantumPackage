from MokeQuantumComputation.Code.QuantumCode.Fermion.FermionicCode import FermionicCode
from MokeQuantumComputation.Code.QuantumCode.Fermion.MajoranaOperator import MajoranaOperator


class FermionicColorCode(FermionicCode):
    def __init__(self,distance):
        super().__init__()
        if distance==3:
            self.define_qubit(7)
            self.push([3, 4, 5, 6], 'x')
            self.push([3, 4, 5, 6], 'z')
            self.push([1, 2, 5, 6], 'x')
            self.push([1, 2, 5, 6], 'z')
            self.push([0, 2, 4, 6], 'x')
            self.push([0, 2, 4, 6], 'z')
            op_x = MajoranaOperator([0, 1, 2, 3, 4, 5, 6], [], 1)
            op_z = MajoranaOperator([],[0, 1, 2, 3, 4, 5, 6], 1)
            self.logical_operator_list_x = [op_x]
            self.logical_operator_list_z = [op_z]
        if distance==5:
            self.define_qubit(19)
            self.push([0,1,2,3], 'x')
            self.push([0,3,5,6], 'x')
            self.push([2,3,4,5,7,8], 'x')
            self.push([5,6,8,9,12,13], 'x')
            self.push([4,7,10,11], 'x')
            self.push([7,8,11,12,15,16], 'x')
            self.push([10,11,14,15], 'x')
            self.push([12,13,16,17], 'x')
            self.push([9,13,17,18], 'x')
            self.push([0,1,2,3], 'z')
            self.push([0,3,5,6], 'z')
            self.push([2,3,4,5,7,8], 'z')
            self.push([5,6,8,9,12,13], 'z')
            self.push([4,7,10,11], 'z')
            self.push([7,8,11,12,15,16], 'z')
            self.push([10,11,14,15], 'z')
            self.push([12,13,16,17], 'z')
            self.push([9,13,17,18], 'z')
            op_x = MajoranaOperator([14,15,16,17,18], [], 1)
            op_z = MajoranaOperator([],[14,15,16,17,18], 1)
            self.logical_operator_list_x = [op_x]
            self.logical_operator_list_z = [op_z]



