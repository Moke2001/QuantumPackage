import copy

from MokeQuantumSimulation.Format.TermFormat.TermFormat import TermFormat


class MultiTermFormat(TermFormat):
    #%%  USER：构造函数
    """""
    MultiTerm类，定义单一多体相互作用量
    self.position_list：list of tuple对象，格点位置列表
    self.op_list：list of str对象，格点上的算符列表
    """""
    def __init__(self, label,effect, position_list, op_list, *args):
        super().__init__(label,effect, *args)
        self.position_list = position_list
        self.op_list = op_list


    # %%  USER：更改算符形式
    def change(self, op_list, *args):
        super().change(*args)
        self.op_list = op_list


    # %%  USER：复制函数
    def copy(self):
        return copy.deepcopy(self)

    
    # %%  KEY：判断参数是否对应MultiTerm
    def fit(self, label,*args):
        if super().__eq__(label):
            if self.position_list == args[0]:
                return True
        return False
    
    
    # %%  KEY：获取算符对象
    def get_op(self):
        return self.op_list
    
    
    #%%  KEY：获取位置对象
    def get_position(self):
        return self.position_list
    
