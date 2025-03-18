import copy
import galois
import numpy as np


class BitString:
    #%%  USER：二进制码的构造函数
    """""
    self.number_bit：int对象，比特数目
    self.bit_vector：np.array of GF(2)对象，二进制串
    形如`np.array([1,0,0,1])`
    """""
    def __init__(self):
        ##  赋值
        self.number_bit=None
        self.bit_vector=None


    #%%  USER：二进制码的第一定义函数
    """""
    input.bit_vector：np.array of GF(2) or list of int对象，二进制码
    influence：给对象赋值
    """""
    def define(self,bit_vector):
        ##  规范处理
        GF = galois.GF(2 ** 1)
        bit_vector_now = GF(np.array(bit_vector.copy(),dtype=int))
        number_bit_now = bit_vector_now.shape[0]

        ##  赋值
        self.bit_vector=bit_vector_now
        self.number_bit=number_bit_now


    #%%  USER：重载加法运算符
    """""
    input.other：BitString对象
    output：BitString对象，按位模二相加的结果
    """""
    def __add__(self,other):
        ##  标准化
        assert isinstance(other, BitString), 'other is not a Codeword'
        assert self.number_bit == other.number_bit, 'number_bits are not equal'

        ##  模二加法
        result=BitString()
        result.define(self.bit_vector+other.bit_vector)
        return result


    #%%  USER：重载减法运算符
    """""
    input.other：BitString对象
    output：BitString对象，按位模二相减的结果
    """""
    def __sub__(self,other):
        ##  标准化
        assert isinstance(other, BitString), 'other is not a Codeword'
        assert self.number_bit == other.number_bit, 'N is not equal to N'

        ##  模二减法
        result=BitString()
        result.define(self.bit_vector - other.bit_vector)
        return result


    #%%  USER：内积运算
    """""
    input.other：BitString对象
    output：int对象，内积结果
    """""
    def inner(self,other):
        assert isinstance(other, BitString), 'other is not a Codeword'
        assert self.number_bit == other.number_bit, 'N is not equal to N'
        result=np.sum(self.bit_vector*other.bit_vector)
        return int(result)


    #%%  USER：复制函数
    """
    output：BitString对象，复制结果
    """
    def copy(self):
        return copy.deepcopy(self)


    #%%  USER：重载等号运算符
    """""
    input.other：BitString对象
    output：bool对象，判断结果
    """""
    def __eq__(self,other):
        return np.all(self.bit_vector == other.bit_vector)


    #%%  USER：获取某一位上的比特信息
    def __getitem__(self, item):
        return self.bit_vector[item]


    #%%  USER：修改某一位上的比特信息
    def __setitem__(self,index,value):
        self.bit_vector[index] = value


    #%%  USER：返回长度信息
    def __len__(self):
        return self.number_bit


    #%%  USER：返回数值信息
    def __str__(self):
        return str(self.bit_vector)


    #%%  USER：求堆积矩阵
    """
    input：list of BitString or np.array of BitString or BitStrings对象，多个二进制码
    output：np.array of GF(2)对象，横排形成的矩阵
    """
    @ staticmethod
    def get_matrix(*args):
        ##  list of BitString or np.array of BitString对象的情况
        if len(args) == 1 and isinstance(args[0],np.ndarray) or isinstance(args[0],list):
            number_bitstring=len(args[0])
            number_bit=args[0][0].number_bit
            matrix = np.zeros((number_bitstring, number_bit), dtype=int)
            for i in range(number_bitstring):
                matrix[i, :] = args[0][i][:]
            GF = galois.GF(2 ** 1)
            matrix=GF(matrix)
            return matrix

        ##  BitStrings对象的情况
        else:
            assert np.all([isinstance(temp,BitString) for temp in args])
            bitstring_vector=np.empty(len(args),dtype=BitString)
            for i in range(len(args)):
                bitstring_vector[i]=args[i]
            return BitString.get_matrix(bitstring_vector)


    #%%  USER：求向量的最大线性无关组
    """
    input：list of BitString or np.array of BitString or BitStrings对象，多个二进制码
    output：np.array of GF(2)对象，线性无关二进制码的数组
    """
    @ staticmethod
    def get_basis(*args):
        ##  list of BitString or np.array of BitString对象的情况
        if len(args) == 1 and isinstance(args[0],np.ndarray) or isinstance(args[0],list):
            GF = galois.GF(2 ** 1)
            number_bitstring=len(args[0])
            number_bit=args[0][0].number_bit
            matrix = BitString.get_matrix(*args)
            rank_origin = np.linalg.matrix_rank(matrix)
            basis_vector=np.empty(rank_origin,dtype=BitString)
            flag=0
            for i in range(number_bitstring):
                matrix[i,:]=GF(np.zeros(number_bit,dtype=int))
                if rank_origin-1==np.linalg.matrix_rank(matrix):
                    basis_vector[flag]=args[0][i]
                    flag+=1
                    rank_origin-=1
            return basis_vector

        ##  BitStrings对象的情况
        else:
            assert np.all([isinstance(temp,BitString) for temp in args])
            bitstring_vector=np.empty(len(args),dtype=BitString)
            for i in range(len(args)):
                bitstring_vector[i]=args[i]
            return BitString.get_basis(bitstring_vector)


    #%%  USER：求向量的秩
    """
    input：list of BitString or np.array of BitString or BitStrings对象，多个二进制码
    output：np.array of GF(2)对象，线性无关二进制码的数组
    """
    @ staticmethod
    def get_rank(*args):
        matrix=BitString.get_matrix(*args)
        return np.linalg.matrix_rank(matrix)


    #%%  USER：求向量的表示
    @staticmethod
    def get_representation(bitstring_vector,bitstring):
        bitstring_vector_merge=np.append(bitstring_vector,bitstring)
        rank_temp=BitString.get_rank(bitstring_vector_merge)
        if rank_temp!=BitString.get_rank(bitstring_vector):
            return None
        zeros=BitString()
        zeros.define(np.zeros(bitstring.number_bit,dtype=int))
        representation=np.zeros(bitstring.number_bit,dtype=int)
        for i in range(len(bitstring_vector)):
            bitstring_vector_merge[i]=zeros
            if BitString.get_rank(bitstring_vector_merge)==rank_temp:
                representation[i]=0
            else:
                representation[i]=1
                rank_temp-=1
        return representation


    @staticmethod
    def independent_judge(*args):
        ##  list of BitString or np.array of BitString对象的情况
        if len(args) == 1 and isinstance(args[0], np.ndarray) or isinstance(args[0], list):
            return len(args[0])==BitString.get_rank(args[0])

        ##  BitStrings对象的情况
        else:
            return len(args)==BitString.get_rank(*args)