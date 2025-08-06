import copy

from MokeQuantumSimulation.Format.TermFormat.TermFormat import TermFormat


class OverallCouplingTermFormat(TermFormat):
    #%%  USER：构造函数
    """""
    OverallCouplingTerm类，定义单一两体相互作用量
    self.cell_index_0：第一个格点所在元胞序号
    self.cell_index_1：第二个格点所在元胞序号
    self.vector：两个元胞的相对格位移
    self.op_0：第一个格点上的算符
    self.op_1：第二个格点上的算符
    """""
    def __init__(self, label,effect, cell_index_0,cell_index_1,cell_vector,op_0,op_1, *args):
        super().__init__(label,effect, *args)
        self.inner_index_0 = cell_index_0
        self.inner_index_1 = cell_index_1
        self.cell_vector = cell_vector
        self.op_0 = op_0
        self.op_1 = op_1


    # %%  USER：更改算符形式
    def change(self, op_0, op_1, *args):
        super().change(*args)
        self.op_0 = op_0
        self.op_1 = op_1


    # %%  UESR：复制函数
    def copy(self):
        return copy.deepcopy(self)

    
    #%%  KEY：判断参数是否对应CouplingTerm
    def fit(self, label,*args):
        if super().fit(label):
            if self.inner_index_0 == args[0] and self.inner_index_1 == args[1] and self.cell_vector == args[2]:
                return True
        return False
    
    
    #%%  KEY：获取算符对象
    def get_op(self):
        return self.op_0,self.op_1
    
    
    #%%  KEY：获取位置对象
    def get_position(self):
        return self.inner_index_0,self.inner_index_1,self.cell_vector
    
