import copy
import numpy as np
from Algebra.BitString.BitString import BitString


class PauliGroup:
    #%%  USER：群构造函数
    """""
    self.number_qubit：int对象，群对应系统的qubit的数目
    self.data_vector：BitString对象，代表site上算符形式的二进制码
    self.coff：int对象，系数
    """""
    def __init__(self):
        self.data_vector = None
        self.number_qubit=None
        self.coff = None


    #%%  USER：群第一定义函数
    """
    input.other：Group对象，乘在右边的群元
    output：Group对象，乘出来的结果
    """
    def define(self,data_vector,coff):
        ##  规范操作
        assert coff==1 or coff==-1 or coff==1j or coff==-1j
        if isinstance(data_vector, np.ndarray) or isinstance(data_vector, list):
            data_vector_now = BitString()
            data_vector_now.define(data_vector)
        elif isinstance(data_vector, BitString):
            data_vector_now = data_vector.copy()
        else:
            raise NotImplementedError('输入参数类型有误')

        ##  赋值
        self.data_vector = data_vector_now
        self.number_qubit = int(data_vector_now.number_bit / 2)
        self.coff = coff


    #%%  USER：重载乘法运算符
    """
    input.other：Group对象，乘在右边的群元
    output：Group对象，乘出来的结果
    """
    def __mul__(self,other):
        ##  如果对方是PauliGroup对象
        if isinstance(other, PauliGroup):
            assert self.number_qubit == other.number_qubit
            ##  计算相乘结果
            coff_new = self.coff * other.coff  # 新算符的系数
            for i in range(int(self.data_vector.number_bit / 2)):
                if self.data_vector[i * 2 + 1]==other.data_vector[i * 2]:
                    coff_new=coff_new*(-1)
            pauli_vector_new= self.data_vector + other.data_vector

            ##  返回结果
            result=PauliGroup()
            result.define(pauli_vector_new,coff_new)
            return result

        ##  如果对方是数，则与系数直接计算
        elif isinstance(other, complex) or isinstance(other, int):
            result=PauliGroup()
            result.define(self.data_vector,self.coff * other)
            return result

        else:
            raise NotImplementedError


    #%%  USER：重载乘法运算符
    """
    input.other：Group对象，乘在左边的群元
    output：Group对象，乘出来的结果
    """
    def __rmul__(self,other):
        if isinstance(other, PauliGroup):
            return other.__mul__(other)
        elif isinstance(other, complex) or isinstance(other, int):
            return self.__mul__(other)


    #%%  USER：重载相等运算符
    """
    input.other：Group对象，与本对象比较的群元
    output：bool对象，判断结果
    """
    def __eq__(self, other):
        assert self.number_qubit == other.number_fermion
        if self.data_vector==other.data_vector and self.coff == other.coff:
            return True
        else:
            return False


    # %%  USER：复制函数
    """
    output：PauliOperator对象，复制结果
    """
    def copy(self):
        return copy.deepcopy(self)


    # %%  USER：算符共轭
    """
    output：PauliOperator对象，共轭结果
    """
    def dag(self):
        if self.coff == 1 or self.coff == -1:
            return self.copy()
        else:
            result = self.copy()
            result.coff = -self.coff
            return result


    # %%  USER：对易式
    """
    input.other：Group对象，判断与本对象是否对易
    output：bool对象，判断结果
    """
    def commuter(self, other):
        flag=True
        for i in range(self.number_qubit):
            if self.data_vector[i * 2 + 1]==other.data_vector[i * 2]==1:
                flag=not flag
            if self.data_vector[i * 2]==other.data_vector[i * 2 + 1]==1:
                flag=not flag
        return flag


    def __getitem__(self,index):
        return self.data_vector[index]


    def __setitem__(self,index,value):
        self.data_vector[index] = value

    #%%  KEY：获取二进制码格式
    @staticmethod
    def get_bitstring(*args):
        if len(args)==1 and (isinstance(args[0],np.ndarray) or isinstance(args[0],list)):
            result=np.empty(len(args[0]),dtype=BitString)
            for i in range(len(args[0])):
                result[i]=args[0][i].data_vector
        else:
            result = np.empty(len(args), dtype=BitString)
            for i in range(len(args)):
                result[i]=args[i].data_vector
        return result


    @staticmethod
    def equal(stabilizer_vector_0,stabilizer_vector_1):
        number_checker_0=stabilizer_vector_0.shape[0]
        number_checker_1=stabilizer_vector_1.shape[0]
        bitstring_vector_0=np.empty(stabilizer_vector_0.shape[0],dtype=BitString)
        bitstring_vector_1=np.empty(stabilizer_vector_1.shape[0],dtype=BitString)
        for i in range(stabilizer_vector_0.shape[0]):
            bitstring_vector_0[i]=stabilizer_vector_0[i].data_vector
        for i in range(stabilizer_vector_1.shape[0]):
            bitstring_vector_1[i]=stabilizer_vector_1[i].data_vector
        rank_0=BitString.get_rank(bitstring_vector_0)
        rank_1=BitString.get_rank(bitstring_vector_1)
        if rank_0!=rank_1:
            return False
        for i in range(number_checker_0):
            representation_temp=BitString.get_representation(bitstring_vector_1,bitstring_vector_0[i])
            if representation_temp is None:
                return False
            eta_temp=1
            for j in range(number_checker_1):
                eta_temp=eta_temp*stabilizer_vector_1[i].coff
            if eta_temp!=stabilizer_vector_0[i].coff:
                return False
        return True