from abc import abstractmethod
import galois
import numpy as np
from Physics.QuantumComputation.Helper.Format import matrix_format
import ldpc.code_util


class QuantumCode:
    def __init__(self):
        self.number_qubit=0
        self.qubit_list=[]
        self.check_list_x=[]
        self.check_list_z=[]
        self.matrix_x=None
        self.matrix_z=None
        self.number_checker_x=0
        self.number_checker_z=0
        self.number_checker=0
        self.rank=None
        self.distance=None
        self.distance_x=None
        self.distance_z=None
        self.number_logical=None
        self.logical_operator_x=[]
        self.logical_operator_z=[]


    def clear(self):
        self.logical_operator_x=[]
        self.logical_operator_z=[]
        self.rank=None
        self.distance=None
        self.number_logical=None


    def define_qubit(self,number):
        self.number_qubit=number
        self.qubit_list=range(self.number_qubit)
        self.clear()

    def define_check(self,matrix_x,matrix_z):
        matrix_x_now=matrix_format(matrix_x)
        matrix_z_now=matrix_format(matrix_z)
        assert matrix_x_now.shape[1]==matrix_z_now.shape[1]==self.number_qubit
        self.matrix_x=matrix_x_now
        self.matrix_z=matrix_z_now
        self.number_checker_x=matrix_x_now.shape[0]
        self.number_checker_z=matrix_z_now.shape[0]
        for i in range(matrix_x_now.shape[0]):
            self.check_list_x.append(np.where(matrix_x_now[i]!=0)[0])
        for i in range(matrix_z_now.shape[0]):
            self.check_list_z.append(np.where(matrix_z_now[i]!=0)[0])
        self.clear()


    def get_matrix(self):
        if self.matrix_x is not None and self.matrix_z is not None:
            return self.matrix_x, self.matrix_z
        else:
            matrix_x=np.zeros((self.number_checker_x,self.number_qubit),dtype=int)
            matrix_z=np.zeros((self.number_checker_z,self.number_qubit),dtype=int)
            for i in range(len(self.check_list_x)):
                matrix_x[i,self.check_list_x[i]]=1
            for i in range(len(self.check_list_z)):
                matrix_z[i,self.check_list_z[i]]=1
            GF=galois.GF(2**1)
            self.matrix_x=GF(matrix_x)
            self.matrix_z=GF(matrix_z)
            return self.matrix_x, self.matrix_z


    def push_qubit(self,number,*args):
        self.number_qubit=number+self.number_qubit
        self.qubit_list=range(self.number_qubit)


    def push_x(self,check):
        assert isinstance(check,np.ndarray)
        self.check_list_x.append(check)
        self.number_checker_x=self.number_checker_x+1
        self.matrix_x=None


    def push_z(self,check):
        assert isinstance(check,np.ndarray)
        self.check_list_z.append(check)
        self.number_checker_z=self.number_checker_z+1
        self.matrix_z=None


    #%%  USER：获取量子纠错码的校验矩阵的秩
    """
    output：int对象，校验矩阵的秩
    influence：修改self.rank为当前求出来的秩
    """
    def get_rank(self):
        if not self.commute_judge():
            raise ValueError('稳定子不对易')
        if self.rank is None:
            rank=np.linalg.matrix_rank(self.matrix_x)+np.linalg.matrix_rank(self.matrix_z)
            self.rank=rank
            self.number_logical= self.number_qubit - rank
            return rank
        else:
            return self.rank


    #%%  USER：获取量子纠错码的逻辑数目
    """
    output：int对象，校验矩阵的逻辑数目
    influence：修改self.rank为当前求出来的秩
    """
    def get_number_logical(self):
        return self.number_qubit-self.get_rank()


    #%%  USER：获取量子纠错码的距离
    """
    output：int对象，量子纠错码的距离
    influence：修改self.distance
    """
    def get_distance(self):
        H_x=np.array(self.matrix_x,dtype=int)
        H_z=np.array(self.matrix_z,dtype=int)
        n_x, k_x, d_x = ldpc.code_util.compute_code_parameters(H_x)
        n_z, k_z, d_z = ldpc.code_util.compute_code_parameters(H_z)
        self.distance_x=d_x
        self.distance_z=d_z
        self.distance=np.min([d_x,d_z])
        return np.min([d_x,d_z])



    #%%  USER：获取量子纠错码的逻辑算符
    """
    output：list of np.array对象，逻辑算符的表示
    influence：修改self.logical_operator
    """
    @abstractmethod
    def get_logical_operators(self):
        pass

    @abstractmethod
    def decoder(self,syndrome_list_x,syndrome_list_z):
        pass

    @abstractmethod
    def commute_judge(self):
        pass