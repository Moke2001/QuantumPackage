import copy
import numpy as np


class PermutationGroup:
    #%%  USER：群构造函数
    """""
    self.number_site：int对象，系统中site的数目
    self.data_vector：np.array of int对象，置换列表
    """""
    def __init__(self):
        self.data_vector = None
        self.number_site=None


    #%%  USER：群第一定义函数
    """
    input.other：Group对象，乘在右边的群元
    output：Group对象，乘出来的结果
    """
    def define(self,data_vector):
        ##  规范操作
        assert isinstance(data_vector,list) or isinstance(data_vector,np.ndarray),'输入参数类型有误'

        ##  赋值
        self.data_vector = np.array(data_vector.copy(),dtype=int)
        self.number_site = self.data_vector.shape[0]


    #%%  USER：重载乘法运算符
    """
    input.other：Group对象，乘在右边的群元
    output：Group对象，乘出来的结果
    """
    def __mul__(self,other):
        assert isinstance(other,PermutationGroup),'参数类型有误'
        assert self.number_site==other.number_site
        data_vector_now=other.data_vector.copy()
        for i in range(self.number_site):
            data_vector_now[i]=self.data_vector[data_vector_now[i]]
        result=PermutationGroup()
        result.define(data_vector_now)
        return result


    #%%  USER：重载乘法运算符
    """
    input.other：Group对象，乘在左边的群元
    output：Group对象，乘出来的结果
    """
    def __rmul__(self,other):
        assert isinstance(other, PermutationGroup), '参数类型有误'
        return other.__mul__(self)


    #%%  USER：重载相等运算符
    """
    input.other：Group对象，与本对象比较的群元
    output：bool对象，判断结果
    """
    def __eq__(self, other):
        assert isinstance(other, PermutationGroup), '参数类型有误'
        if np.all(self.data_vector == other.data_vector):
            return True
        else:
            return False


    # %%  USER：复制函数
    """
    output：PauliOperator对象，复制结果
    """
    def copy(self):
        return copy.deepcopy(self)


    def __getitem__(self,index):
        return self.data_vector[index]


    def __setitem__(self,index,value):
        self.data_vector[index] = value