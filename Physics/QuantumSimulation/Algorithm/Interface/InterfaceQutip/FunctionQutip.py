from qutip import *


class  FunctionQutip:
    # %%  KEY：用于qutip时变函数参数处理的函数类构造函数
    """"
    self.function：函数句柄对象，时变函数
    self.function_params：字典对象，指定参数
    """""
    def __init__(self,function,function_params):
        self.function=function
        self.function_params=function_params


    #%%  KEY：作为qutip计算时函数句柄的函数
    """
    input.t：float对象，时间
    output：float or complex or int对象，时变函数根据时间和指定参数的返回值
    """
    def MyFunction(self,t,*args):
        return self.function(t,self.function_params)
