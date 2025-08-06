import copy
import numpy as np
from tenpy import MPS
from MokeQuantumSimulation.Algorithm.Interface.InterfaceTenpy.GetLatticeTenpy import *
from MokeQuantumSimulation.State.StateNumpy.StateNumpy import StateNumpy


class StateTenpy:
    # %%  USER：构造函数
    def __init__(self,mps,dimension_array):
        self.mps=mps
        self.dimension_array=dimension_array


    # %%  USER：重载加法
    def __add__(self, other):
        assert isinstance(other, StateTenpy)
        return StateTenpy(self.mps.add(other.mps,1,1),self.dimension_array)


    # %%  USER：重载减法
    def __sub__(self, other):
        assert isinstance(other,StateTenpy)
        return StateTenpy(self.mps.add(other.mps,1,-1),self.dimension_array)


    # %%  USER：重载数乘
    def __mul__(self, other):
        assert isinstance(other, complex) or isinstance(other, float) or isinstance(other, int)
        return StateTenpy(self.mps.add(self.mps,0,other),self.dimension_array)


    # %%  USER：重载数乘
    def __rmul__(self, other):
        return self.__mul__(other)


    # %%  USER：重载除法
    def __truediv__(self, other):
        assert isinstance(other, complex) or isinstance(other, float) or isinstance(other, int)
        return StateTenpy(self.mps.add(self.mps,0,1/other),self.dimension_array)


    # %%  USER：计算模长
    def norm(self):
        return self.mps.norm


    # %%  USER：归一化
    def normalize(self):
        self.mps.canonical_form()


    # %%  KEY：复制函数
    def copy(self):
        return copy.deepcopy(self)


    #%% USER：根据StateNumpy构造
    @staticmethod
    def FromStateNumpy(state_numpy):
        assert isinstance(state_numpy,StateNumpy)
        site_number=np.prod(state_numpy.dimension_array.shape)
        cell_period_list=list(state_numpy.dimension_array.shape[0:-1])
        index_tuple=()
        for i in range(len(cell_period_list)):
            index_tuple=index_tuple+(cell_period_list[i]-1,)
        site_dimension_list=list(state_numpy.dimension_array[index_tuple])
        site_tenpy_list=[]
        for i in range(len(site_dimension_list)):
            ##  根据维度定义Site对象
            site_temp = Site(LegCharge.from_trivial(site_dimension_list[i]))
            site_tenpy_list.append(site_temp)
        lattice = Lattice(cell_period_list, site_tenpy_list)
        leg_list = [None] * site_number
        label_list = ['none'] * site_number
        for index_temp, ignore in np.ndenumerate(state_numpy.dimension_array):
            mps_index = lattice.lat2mps_idx(index_temp)
            leg_list[mps_index] = LegCharge.from_trivial(site_dimension_list[index_temp[-1]])
            label_list[mps_index] = 'p' + str(mps_index)
        temp = Array.from_ndarray(state_numpy.tensor, legcharges=leg_list, labels=label_list)
        mps = MPS.from_full(lattice.mps_sites(), temp)
        return StateTenpy(mps,state_numpy.dimension_array.copy())




