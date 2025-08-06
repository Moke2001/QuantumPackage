import copy
from MokeQuantumSimulation.Format.LatticeFormat.LatticeFormat import LatticeFormat
from MokeQuantumSimulation.Format.TermFormat.TermsFormat import TermsFormat


class ModelFormat(LatticeFormat,TermsFormat):
    # %%  USER：构造函数
    """""
    ModelFormat：格式类模型对象
    self.cell_period_list：元胞延展次数，形如[1,2,3]
    self.cell_vector_list：每个方向上的元胞基矢，形如[np.array,np.array]
    self.inner_site_list：元胞内Site对象列表，形如[Site,Site]
    self.inner_coordinate：元胞内格点位移，形如[np.array,np.array]，与cell_sites对应
    self.cell_number：元胞个数
    self.space_dimension：空间维度数目
    self.site_number_in_one_cell：元胞中格点个数
    self.terms：list of Term对象，储存模型的作用量
    self.number：int对象，储存作用量个数
    output：ModelFormat对象
    influence：本函数不改变参数对象
    """""
    def __init__(self,conserve=None):
        LatticeFormat.__init__(self)
        TermsFormat.__init__(self)
        self.conserve=conserve


    #%%  KEY：复制函数
    """""
    output：ModelFormat对象
    influence：本函数不改变参数对象
    """""
    def copy(self):
        return copy.deepcopy(self)