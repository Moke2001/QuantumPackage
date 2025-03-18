import ldpc.code_util
import numpy as np
from Physics.QuantumComputation.Helper.Format import matrix_format


class LinearCode:
    #%%  USER：构造函数
    """""
    self.number_checker：int对象，校验子的数目
    self.number_bit：int对象，bit的数目
    self.check_matrix：np.array of GF(2)对象，校验矩阵
    self.rank：int对象，校验矩阵的秩
    self.distance：int对象，线性码的码距
    self.codeword_matrix：np.array of GF(2)对象，生成矩阵
    """""
    def __init__(self,check_matrix):
        check_matrix_now=matrix_format(check_matrix)
        self.number_checker=check_matrix_now.shape[0]
        self.number_bit=check_matrix_now.shape[1]
        self.check_matrix=check_matrix_now
        self.rank=None
        self.distance=None
        self.codeword_matrix=None


    #%%  USER：求码字
    """
    output：np.array of GF(2)对象，生成矩阵
    """
    def get_codewords(self):
        if self.codeword_matrix is not None:
            return self.codeword_matrix
        else:
            codeword_matrix = self.check_matrix.null_space()
            self.codeword_matrix=codeword_matrix
            with open("codespace.txt", "w") as f:
                f.write("matrix=" + str(codeword_matrix.tolist()))
            return codeword_matrix


    #%%  USER：求对偶码
    """
    output：LinearCode对象，对偶码
    """
    def get_dual_code(self):
        return LinearCode(self.get_codewords())


    #%%  USER：获取最大可纠错的距离
    """
    output：int对象，码距
    """
    def get_distance(self):
        n_x, k_x, d_x = ldpc.code_util.compute_code_parameters(np.array(self.check_matrix,dtype=int))
        return d_x