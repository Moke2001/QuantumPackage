import copy
import numpy as np


class StateNumpy:
    #%%  USER：构造函数
    def __init__(self,tensor,dimension_array):
        self.tensor=tensor
        self.dimension_array=dimension_array


    #%%  USER：重载加法
    def __add__(self,other):
        assert isinstance(other,StateNumpy)
        assert self.dimension_array==other.dimension_array
        return StateNumpy(self.tensor + other.tensor,self.dimension_array)


    # %%  USER：重载减法
    def __sub__(self,other):
        assert isinstance(other,StateNumpy)
        assert self.dimension_array==other.dimension_array
        return StateNumpy(self.tensor - other.tensor,self.dimension_array)


    # %%  USER：重载数乘
    def __mul__(self,other):
        assert isinstance(other,complex) or isinstance(other,float) or isinstance(other,int)
        return StateNumpy(self.tensor * other, self.dimension_array)


    # %%  USER：重载数乘
    def __rmul__(self,other):
        return self.__mul__(other)


    # %%  USER：重载数除
    def __truediv__(self, other):
        assert isinstance(other, complex) or isinstance(other, float) or isinstance(other, int)
        return StateNumpy(self.tensor / other, self.dimension_array)


    # %%  KEY：复制函数
    def copy(self):
        return copy.deepcopy(self)


    #%%  USER：基于局域态矢构造numpy态矢
    @staticmethod
    def FromLocalState(*args):
        assert all(isinstance(it,np.ndarray) for it in args)
        dimension_array=np.empty(args[0].shape[0:-1])
        for index_tuple, amount_temp in np.ndenumerate(args[0][...,0]):
            dimension_array[index_tuple]=args[0][index_tuple].shape[0]
        tensor=None
        for i in range(len(args)):
            tensor_temp=None
            for index_tuple, amount_temp in np.ndenumerate(args[i][...,0]):
                if tensor_temp is None:
                    tensor_temp=np.array(args[i][index_tuple])
                else:
                    tensor_temp=np.einsum('...,i->...i',tensor_temp,np.array(args[i][index_tuple]))
            if tensor is None:
                tensor=tensor_temp
            else:
                tensor=tensor+tensor_temp
        return StateNumpy(tensor,dimension_array)
