import galois
import numpy as np


class PauliOperator:

    #%%  USER：构造函数
    """""
    self.x_vector：np.array of int对象，占位序号
    self.z_vector：np.array of int对象，占位序号
    self.coff：int对象，系数符号
    self.factor：complex对象，系数虚实
    """""
    def __init__(self,x_vector,z_vector,coff):

        ##  数据标准化
        assert isinstance(x_vector,np.ndarray) or isinstance(x_vector,list)
        assert isinstance(z_vector,np.ndarray) or isinstance(z_vector,list)
        assert coff==-1 or coff==1 or coff==1j or coff==-1j

        ##  赋值
        self.x_vector=np.array(x_vector,dtype=int)
        self.z_vector=np.array(z_vector,dtype=int)
        self.coff=coff


    #%%  USER：将序号推后
    """
    input.number：int对象，序号退后的个数
    influence：修改self.x_vector和self.z_vector
    """
    def shift(self,number):
        self.x_vector=self.x_vector+number
        self.z_vector=self.z_vector+number


    #%%  USER：计算对偶算符
    """
    output：MajoranaOperator对象，对偶算符
    """
    def dual(self):
        return PauliOperator(self.z_vector,self.x_vector,self.coff)


    #%%  KEY：计算矩阵形式
    """
    input.number_qubit：int对象，总共的qubits数目
    output：np.array of int对象，校验子向量
    """
    def get_matrix(self,number_qubit):
        GF=galois.GF(2**1)
        matrix=GF(np.zeros(number_qubit*2, dtype=int))
        matrix[self.x_vector*2]=1
        matrix[self.z_vector*2+1]=1
        return matrix


    #%%  USER：判断两个算符是否对易
    """
    input.other：MajoranaOperator对象，另一个算符
    output：bool对象，判断结果
    """
    def commute(self,other):
        assert isinstance(other,PauliOperator)
        overlap_x=len(np.intersect1d(self.x_vector,other.z_vector))
        overlap_z=len(np.intersect1d(self.z_vector,other.x_vector))
        judge=overlap_x+overlap_z
        return np.mod(judge,2)==0


    #%%  USER：计算两个算符相乘的结果
    """
    input.other：MajoranaOperator对象，另一个算符
    input.number_qubit：int对象，总共的qubits数目
    output：MajoranaOperator对象，相乘的结果
    """
    def mul(self,other,number_qubit):
        assert isinstance(other,PauliOperator)
        ##  计算相乘结果
        GF=galois.GF(2**1)
        data_0=GF(np.zeros(number_qubit*2,dtype=int))
        data_1=GF(np.zeros(number_qubit*2,dtype=int))
        data_0[self.x_vector*2]=1
        data_0[self.z_vector*2+1]=1
        data_1[other.x_vector*2]=1
        data_1[other.z_vector*2+1]=1

        data_new = data_0+data_1
        if self.commute(other):
            new_coff = self.coff * other.coff
        else:
            new_coff = self.coff * other.coff
        x_vector = np.where(data_new[0::2] == 1)[0]
        z_vector = np.where(data_new[1::2] == 1)[0]
        return PauliOperator(x_vector,z_vector,new_coff)


    #%%  USER：复制函数
    """
    output：MajoranaOperator对象，复制结果
    """
    def copy(self):
        return PauliOperator(self.x_vector,self.z_vector,self.coff)


    #%%  KEY：重构输出函数
    """
    output：tuple对象，占位向量和系数
    """
    def __str__(self):
        return str('x')+str(self.x_vector)+str('z')+str(self.z_vector)+str('coff')+str(self.coff)




