import copy

from Physics.QuantumSimulation.Format.TermFormat.TermFormat import TermFormat


class OverallMultiTermFormat(TermFormat):
    #%%  USER：构造函数
    """""
    OverallMultiTerm类，定义单一两体相互作用量
    self.cell_index_list：格点在元胞中序号列表
    self.vector_list：格位移列表
    self.op_list：算符列表
    """""
    def __init__(self, label,effect, cell_index_list, vector_list, op_list, *args):
        super().__init__(label,effect, *args)
        self.cell_index_list = cell_index_list
        self.vector_list = vector_list
        self.op_list = op_list


    # %%  USER：更改算符形式
    def change(self, op_list, *args):
        super().change(*args)
        self.op_list = op_list


    # %%  USER：复制函数
    def copy(self):
        return copy.deepcopy(self)


    # %%  KEY：判断参数是否对应CouplingTerm
    def fit(self, label, *args):
        if super().fit(label):
            if self.cell_index_list == args[0] and self.vector_list == args[1]:
                return True
        return False


    # %%  KEY：获取算符对象
    def get_op(self):
        return self.op_list


    # %%  KEY：获取位置对象
    def get_position(self):
        return self.cell_index_list, self.vector_list
