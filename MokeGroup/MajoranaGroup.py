import copy
import numpy as np
from Algebra.BitString.BitString import BitString


class MajoranaGroup:
    #%%  USER：构造函数
    """""
    Majorana Group的表示，这个类包含了所有参数N不同的Majorana group中的元素
    self.number_qubit：int对象，Fermionic sites的数目
    self.majorana_vector：np.array of int对象，用于表示对应位上算符存在性的一阶张量
    例如[1,0,0,1]代表了$\hat\gamma_0\hat\gamma_1'$
    self.coff：complex对象，只能为1，-1，1j，-1j四个数值的系数
    """""
    def __init__(self):
        self.data_vector = None
        self.coff = None
        self.number_fermion=None


    # %%  USER：群第一定义函数
    """
    input.other：Group对象，乘在右边的群元
    output：Group对象，乘出来的结果
    """
    def define(self,data_vector,coff):
        ##  规范操作
        assert coff == 1 or coff == -1 or coff == 1j or coff == -1j
        if isinstance(data_vector, np.ndarray) or isinstance(data_vector, list):
            data_vector_now = BitString()
            data_vector_now.define(data_vector)
        elif isinstance(data_vector, BitString):
            data_vector_now = data_vector.copy()
        else:
            raise NotImplementedError('输入参数类型有误')

        ##  赋值
        self.data_vector = data_vector_now
        self.number_fermion = int(data_vector_now.number_bit / 2)
        self.coff = coff


    #%%  USER：重载乘法运算符
    """
    input.other：MajoranaGroup对象或complex对象或int对象，右边乘上的对象
    output：MajoranaGroup对象，乘出来的结果
    """
    def __mul__(self,other):
        ##  如果对方是MajoranaGroup对象
        if isinstance(other, MajoranaGroup):
            assert len(self.data_vector) == len(self.data_vector)  # 只有同群才能相乘

            ##  计算相乘结果
            majorana_vector_new = np.zeros_like(self.data_vector, dtype=int)
            new_coff = self.coff * other.coff

            """
            如果对方的算符是1，将对方的算符依次替换到前面，直到到达对应位置上
            替换过程中如果经过一个1，符号要发生一次变化
            如果对方的算符是0，那么不需要替换
            """
            for i in range(len(other.data_vector)):
                if other.data_vector[i] == 0:
                    majorana_vector_new[i] = self.data_vector[i]
                else:
                    for j in range(len(self.data_vector) - 1, i, -1):
                        if self.data_vector[i] == 1:
                            new_coff = -new_coff
                    if self.data_vector[i] == 0:
                        majorana_vector_new[i]=1
                    else:
                        majorana_vector_new[i]=0
            result=MajoranaGroup()
            result.define(majorana_vector_new,new_coff)
            return

        ##  如果对方是数，则与系数直接计算
        elif isinstance(other, complex) or isinstance(other, int):
            result = self.copy()
            result.coff = self.coff * other
            return result


    # %%  USER：重载乘法运算符
    """
    input.other：MajoranaGroup对象或complex对象或int对象，右边乘上的对象
    output：MajoranaGroup对象，乘出来的结果
    """
    def __rmul__(self,other):
        if isinstance(other, MajoranaGroup):
            return other.__mul__(other)
        elif isinstance(other, complex) or isinstance(other, int):
            return self.__mul__(other)


    # %%  USER：重载等于号运算符
    """
    input.other：any对象，只有MajoranaGroup对象才有可能相等
    output：bool对象，判断结果
    """
    def __eq__(self, other):
        if isinstance(other, MajoranaGroup):
            if self.number_fermion==other.number_fermion:
                if np.all(self.data_vector == other.data_vector):
                    return True
        return False


    # %%  USER：复制函数
    """
    output：MajoranaGroup对象，复制结果
    """
    def copy(self):
        return copy.deepcopy(self)


    # %%  USER：算符共轭
    """
    output：MajoranaOperator对象，共轭结果
    """
    def dag(self):
        num = len([item for item in self.data_vector if item == 1])
        factor = (-1)**(num * (num - 1) / 2)
        if self.coff==1 or self.coff==-1:
            result = self.copy()
            result.coff=result.coff*factor
            return result
        else:
            result=self.copy()
            result.coff=-result.coff*factor
            return result


    #%%  USER：计算权重
    """
    output：int对象，含有1的个数
    """
    def get_weight(self):
        weight=0
        for i in range(self.number_fermion * 2):
            weight=weight+int(self.data_vector[i])
        return weight


    # %%  USER：对易式
    """
    output：bool对象，是否对易的判断
    """
    def commuter(self, other):
        ##  标准化步骤
        assert isinstance(other, MajoranaGroup), 'Other参数必须是MajoranaOperator对象'
        assert (self.number_fermion == other.number_fermion), '稳定子字符串长度必须与sites数目相同'

        ##  根据对易关系计算两个算符之间的对易性
        result=0
        result=result+self.data_vector.inner(other.data_vector)
        factor=result+self.get_weight()*other.get_weight()

        ##  返回结果
        return np.mod(factor,2)==0


    # %%  USER：获取第item个位置上的算符
    def __getitem__(self, item):
        return self.data_vector[item]


    # %%  USER：设置第item位置上的算符
    def __setitem__(self, key, value):
        self.data_vector[key] = value


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