import copy
import numpy as np
from qutip import Qobj
from MokeQuantumSimulation.State.StateNumpy.StateNumpy import StateNumpy


class StateQutip:
    #%%  USER：构造函数
    def __init__(self,vector,dimension_array):
        self.vector=vector
        self.dimension_array=dimension_array


    # %%  USER：重载加法
    def __add__(self, other):
        assert isinstance(other, StateQutip)
        return StateQutip(self.vector+other.vector,self.dimension_array)


    # %%  USER：重载减法
    def __sub__(self, other):
        assert isinstance(other,StateQutip)
        return StateQutip(self.vector-other.vector,self.dimension_array)


    # %%  USER：重载数乘
    def __mul__(self, other):
        assert isinstance(other, complex) or isinstance(other, float) or isinstance(other, int)
        return StateQutip(self.vector * other, self.dimension_array)


    # %%  USER：重载数乘
    def __rmul__(self, other):
        return self.__mul__(other)


    # %%  USER：重载数除
    def __truediv__(self, other):
        assert isinstance(other, complex) or isinstance(other, float) or isinstance(other, int)
        return StateQutip(self.vector*(1/other), self.dimension_array)


    # %%  USER：计算模长
    def norm(self):
        return self.vector.norm()


    # %%  USER：归一化
    def normalize(self):
        self.vector=self.vector/self.norm()


    # %%  KEY：复制函数
    def copy(self):
        return copy.deepcopy(self)


    #%%  USER：根据StateNumpy构造
    @staticmethod
    def FromStateNumpy(state_numpy):
        assert isinstance(state_numpy,StateNumpy)
        tensor=state_numpy.tensor
        vector = Qobj(tensor.reshape((np.prod(tensor.shape), 1)), dims=[list(tensor.shape),[1] * len(tensor.shape)])
        dimension_array = state_numpy.dimension_array
        return StateQutip(vector, dimension_array)
