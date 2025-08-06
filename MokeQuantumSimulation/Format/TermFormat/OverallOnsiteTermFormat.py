import copy

from MokeQuantumSimulation.Format.TermFormat.TermFormat import TermFormat


class OverallOnsiteTermFormat(TermFormat):
    #%%  USER：构造函数
    """""
    OverallOnsiteTerm类，定义单一两体相互作用量
    self.inner_index：格点在元胞中的序号
    self.op：格点上的算符
    """""
    def __init__(self, label,effect, cell_index,op, *args):
        super().__init__(label,effect, *args)
        self.inner_index = cell_index
        self.op = op


    # %%  USER：更改算符形式
    def change(self, op, *args):
        super().change(*args)
        self.op = op


    # %%  USER：复制函数
    def copy(self):
        return copy.deepcopy(self)
    
    
    #%%  KEY：判断参数是否对应CouplingTerm
    def fit(self, label,*args):
        if super().fit(label):
            if self.inner_index == args[0]:
                return True
        return False
    
    
    #%%  KEY：获取算符对象
    def get_op(self):
        return self.op
    
    
    #%%  KEY：获取位置对象
    def get_position(self):
        return self.inner_index
    
