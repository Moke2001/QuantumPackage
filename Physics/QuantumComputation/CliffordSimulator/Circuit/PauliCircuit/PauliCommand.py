


class PauliCommand:
    #%%  USER：构造函数
    """""
    self.index：int对象，作用目标site的序号
    self.control_index：int对象，控制位site的序号
    self.target_index：int对象，目标位site的序号
    self.name：str对象，指令名称
    self.stamp：object对象，时间戳
    """""
    def __init__(self):
        self.index=None
        self.control_index=None
        self.target_index=None
        self.name=None
        self.stamp=None


    #%%  USER：定义函数
    """""
    input.index：int对象，作用目标site的序号
    input.control_index：int对象，控制位site的序号
    input.target_index：int对象，目标位site的序号
    input.name：str对象，指令名称
    input.stamp：object对象，时间戳
    influence：修改self属性
    """""
    def define(self,gate_name,index,control_index,target_index,stamp):
        self.index=index
        self.target_index=target_index
        self.control_index=control_index
        self.name=gate_name
        self.stamp=stamp


    #%%  USER：生成X门的指令条目
    """""
    input.index：int对象，作用目标site的序号
    input.args：object对象，可能存在的时间戳
    output：PauliCommand对象，产生的指令条目
    """""
    @staticmethod
    def X(index,*args):
        if len(args)==1:
            stamp=args[0]
        else:
            stamp=None
        gate=PauliCommand()
        gate.define('X',index,None,None,stamp)
        return gate


    #%%  USER：生成Y门的指令条目
    """""
    input.index：int对象，作用目标site的序号
    input.args：object对象，可能存在的时间戳
    output：PauliCommand对象，产生的指令条目
    """""
    @staticmethod
    def Y(index,*args):
        if len(args)==1:
            stamp=args[0]
        else:
            stamp=None
        gate=PauliCommand()
        gate.define('Y',index,None,None,stamp)
        return gate


    #%%  USER：生成Z门的指令条目
    """""
    input.index：int对象，作用目标site的序号
    input.args：object对象，可能存在的时间戳
    output：PauliCommand对象，产生的指令条目
    """""
    @staticmethod
    def Z(index,*args):
        if len(args)==1:
            stamp=args[0]
        else:
            stamp=None
        gate=PauliCommand()
        gate.define('Z',index,None,None,stamp)
        return gate


    #%%  USER：生成S门的指令条目
    """""
    input.index：int对象，作用目标site的序号
    input.args：object对象，可能存在的时间戳
    output：PauliCommand对象，产生的指令条目
    """""
    @staticmethod
    def S(index,*args):
        if len(args)==1:
            stamp=args[0]
        else:
            stamp=None
        gate=PauliCommand()
        gate.define('S',index,None,None,stamp)
        return gate


    #%%  USER：生成H门的指令条目
    """""
    input.index：int对象，作用目标site的序号
    input.args：object对象，可能存在的时间戳
    output：PauliCommand对象，产生的指令条目
    """""
    @staticmethod
    def H(index,*args):
        if len(args)==1:
            stamp=args[0]
        else:
            stamp=None
        gate=PauliCommand()
        gate.define('H',index,None,None,stamp)
        return gate


    #%%  USER：生成CZ门的指令条目
    """""
    input.index_control：int对象，控制位site的序号
    input.index_target：int对象，目标位site的序号
    input.args：object对象，可能存在的时间戳
    output：PauliCommand对象，产生的指令条目
    """""
    @staticmethod
    def CX(control_index,target_index,*args):
        if len(args)==1:
            stamp=args[0]
        else:
            stamp=None
        gate=PauliCommand()
        gate.define('CX',None,control_index,target_index,stamp)
        return gate


    #%%  USER：生成CY门的指令条目
    """""
    input.index_control：int对象，控制位site的序号
    input.index_target：int对象，目标位site的序号
    input.args：object对象，可能存在的时间戳
    output：PauliCommand对象，产生的指令条目
    """""
    @staticmethod
    def CY(control_index, target_index,*args):
        if len(args)==1:
            stamp=args[0]
        else:
            stamp=None
        gate = PauliCommand()
        gate.define('CY', None, control_index, target_index,stamp)
        return gate


    #%%  USER：生成CZ门的指令条目
    """""
    input.index_control：int对象，控制位site的序号
    input.index_target：int对象，目标位site的序号
    input.args：object对象，可能存在的时间戳
    output：PauliCommand对象，产生的指令条目
    """""
    @staticmethod
    def CZ(control_index,target_index,*args):
        if len(args)==1:
            stamp=args[0]
        else:
            stamp=None
        gate=PauliCommand()
        gate.define('CZ',None,control_index,target_index,stamp)
        return gate


    #%%  USER：生成SWAP门的指令条目
    """""
    input.index_control：int对象，控制位site的序号
    input.index_target：int对象，目标位site的序号
    input.args：object对象，可能存在的时间戳
    output：PauliCommand对象，产生的指令条目
    """""
    @staticmethod
    def SWAP(control_index,target_index,*args):
        if len(args)==1:
            stamp=args[0]
        else:
            stamp=None
        gate=PauliCommand()
        gate.define('SWAP',None,control_index,target_index,stamp)
        return gate


    #%%  USER：生成测量的指令条目
    """""
    input.index：int对象，测量目标qubit的序号
    input.args：object对象，可能存在的时间戳
    output：PauliCommand对象，产生的指令条目
    """""
    @staticmethod
    def measure(index,*args):
        if len(args)==1:
            stamp=args[0]
        else:
            stamp=None
        gate=PauliCommand()
        gate.define('M',index,None,None,stamp)


