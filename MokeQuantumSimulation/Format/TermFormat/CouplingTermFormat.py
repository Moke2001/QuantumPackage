import copy
from MokeQuantumSimulation.Format.TermFormat.TermFormat import TermFormat


class CouplingTermFormat(TermFormat):
    #%%  USER：构造函数
    """""
    CouplingTerm类，定义单一两体相互作用量
    self.position_0：第一个格点位置
    self.position_1：第二个格点位置
    self.op_0：第一个格点上的算符
    self.op_1：第二个格点上的算符
    """""
    def __init__(self, label,effect, position_0, position_1, op_0, op_1, *args):
        ##  SECTION：标准化-------------------------------------------------------------------------
        assert isinstance(label, str),'参数label必须是str对象'
        assert isinstance(effect,str),'参数effect必须是str对象'
        assert isinstance(position_0,tuple),'参数position_0必须是tuple对象'
        assert isinstance(position_1, tuple), '参数position_1必须是tuple对象'
        assert isinstance(op_0,str),'参数op_0必须是str对象'
        assert isinstance(op_1, str), '参数op_1必须是str对象'

        ##  SECTION：赋值---------------------------------------------------------------------------
        super().__init__(label,effect, *args)
        self.position_0 = position_0
        self.position_1 = position_1
        self.op_0 = op_0
        self.op_1 = op_1


    #%%  USER：更改算符形式
    def change(self, op_0, op_1, *args):
        super().change(*args)
        self.op_0 = op_0
        self.op_1 = op_1


    #%%  USER：复制函数
    def copy(self):
        return copy.deepcopy(self)


    #%%  KEY：判断参数是否对应CouplingTerm
    def fit(self, label, *args):
        if super().__eq__(label):
            if len(args) >= 2:
                if self.position_0 == args[0] and self.position_1 == args[1]:
                    return True
        return False


    #%%  KEY：获取算符对象
    def get_op(self):
        return self.op_0, self.op_1


    #%%  KEY：获取位置对象
    def get_position(self):
        return self.position_0, self.position_1
