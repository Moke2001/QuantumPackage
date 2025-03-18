import numpy as np


class LatticeFormat:
    # %%  USER：构造函数
    """""
    LatticeFormat：格式晶格类
    self.cell_period_list：元胞延展次数，形如[1,2,3]
    self.cell_vector_list：每个方向上的元胞基矢，形如[np.array,np.array]
    self.inner_site_list：元胞内Site对象列表，形如[Site,Site]
    self.inner_coordinate：元胞内格点位移，形如[np.array,np.array]，与cell_sites对应
    self.cell_number：元胞个数
    self.space_dimension：空间维度数目
    self.site_number_in_one_cell：元胞中格点个数
    """""
    def __init__(self):
        self.cell_period_list = None
        self.cell_vector_list = None
        self.site_dimension_list=None
        self.site_vector_list=None
        self.periodicity = None
        self.local_operator_dictionary=None
        self.initialize_local_operator_dictionary()


    #%%  KEY：固定局域算符
    def initialize_local_operator_dictionary(self):
        self.local_operator_dictionary= \
            {'sigmax'    : np.array([[0, 1], [1, 0]], dtype=complex),
            'sigmay'    : np.array([[0, -1j], [1j, 0]], dtype=complex),
            'sigmaz'    : np.array([[1, 0], [0, -1]], dtype=complex),
            'sigmaup'   : np.array([[1, 0], [0, 0]], dtype=complex),
            'sigmadown' : np.array([[0, 0], [0, 1]], dtype=complex),
            'sigmaplus' : np.array([[0, 1], [0, 0]], dtype=complex),
            'sigmaminus': np.array([[0, 0], [1, 0]], dtype=complex)}


    #%%  USER：定义局域算符
    def add_local_operator(self,name,operator_numpy):
        self.local_operator_dictionary[name] = operator_numpy


    #%%  USER：定义晶格周期
    def set_period(self,cell_period_list):
        assert isinstance(cell_period_list,list),'参数Ls必须是list对象'
        assert all(isinstance(it,int) for it in cell_period_list),'参数Ls元素必须是int对象'
        self.cell_period_list=cell_period_list


    #%%  USER：定义晶格元胞基矢
    def set_cell_vector(self,cell_vector_list):
        assert isinstance(cell_vector_list,list)
        assert all(isinstance(it,np.ndarray) for it in cell_vector_list)
        self.cell_vector_list=cell_vector_list


    #%%  USER：定义元胞内格点Hilbert空间维度
    def set_site_dimension(self,site_dimension_list):
        assert isinstance(site_dimension_list,list)
        assert all(isinstance(it,int) for it in site_dimension_list)
        self.site_dimension_list=site_dimension_list


    #%%  USER：定义元胞内格点相对于元胞的位移
    def set_site_vector(self, inner_vector_list):
        assert isinstance(inner_vector_list,list)
        assert all(isinstance(it,np.ndarray) for it in inner_vector_list)
        self.site_vector_list=inner_vector_list


    #%%  USER：定义周期性
    def set_periodicity(self,periodicity):
        if isinstance(periodicity,bool):
            self.periodicity=[periodicity]*len(self.cell_period_list)
        elif isinstance(periodicity,list):
            self.periodicity=periodicity
        else:
            raise TypeError('参数periodicity类型错误')


    # %%  USER：获得格点坐标
    """""
    input.index_tuple：格坐标
    output：格点空间坐标
    influence：本函数不改变参数对象
    """""
    def get_position(self, index_tuple):
        result = np.zeros(len(self.cell_vector_list))
        for i in range(len(self.cell_vector_list)):
            result = result + self.cell_vector_list[i] * index_tuple[i]
        result = result + self.site_vector_list[index_tuple[-1]]
        return result


    # %%  USER：求两个Site间的距离
    """""
    input.index_tuple_0：前一个格坐标
    input.index_tuple_1：后一个格坐标
    output：float对象，格点距离
    influence：本函数不改变参数对象
    """""
    def get_distance(self, index_tuple_0, index_tuple_1):
        position_0 = self.get_position(index_tuple_0)
        position_1 = self.get_position(index_tuple_1)
        return np.linalg.norm(position_0 - position_1)


    #%%  USER：返回晶格局域维度数组
    @property
    def dimension_array(self):
        array=np.empty(self.cell_period_list + [len(self.site_dimension_list)], dtype=int)
        for index_tuple_temp, object_temp in np.ndenumerate(array):
            array[index_tuple_temp] = self.site_dimension_list[index_tuple_temp[-1]]
        return array
