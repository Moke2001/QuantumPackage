from abc import abstractmethod
import galois
import numpy as np
from ldpc import BpOsdDecoder

from Physics.QuantumComputation.Code.QuantumCode.QuantumCode import QuantumCode


class PauliCode(QuantumCode):
    def __init__(self):
        super().__init__()


    #%%  USER：获取量子纠错码的逻辑算符
    """
    output：list of np.array对象，逻辑算符的表示
    influence：修改self.logical_operator
    """
    @abstractmethod
    def get_logical_operators(self):
        if self.logical_operator_x!=[] and self.logical_operator_z!=[]:
            return self.logical_operator_x,self.logical_operator_z
        self.get_matrix()
        codewords_x = self.matrix_x.null_space()
        codewords_z = self.matrix_z.null_space()
        GF=galois.GF(2**1)
        matrix_x=self.matrix_x.copy()
        matrix_z=self.matrix_z.copy()

        # 筛选与行空间无关的基矢
        independent_null_basis_list_0 = []
        for vec in codewords_x:
            rank_before = np.linalg.matrix_rank(matrix_x)
            matrix_x=np.vstack([matrix_x,GF(np.array(vec, dtype=int))])
            if np.linalg.matrix_rank(matrix_x) == rank_before + 1:
                independent_null_basis_list_0.append(vec)
            else:
                matrix_x=np.delete(matrix_x,-1,axis=0)
        independent_null_basis_list_1 = []
        for vec in codewords_z:
            rank_before = np.linalg.matrix_rank(matrix_z)
            matrix_z=np.vstack([matrix_z,GF(np.array(vec, dtype=int))])
            if np.linalg.matrix_rank(matrix_z) == rank_before + 1:
                independent_null_basis_list_1.append(vec)
            else:
                matrix_z=np.delete(matrix_z,-1,axis=0)

        ##  加入逻辑算符中
        for vec in independent_null_basis_list_1:
            self.logical_operator_x.append(np.where(vec!=0)[0])
        for vec in independent_null_basis_list_0:
            self.logical_operator_z.append(np.where(vec!=0)[0])

        ##  返回结果
        return self.logical_operator_x,self.logical_operator_z


    def decoder(self, syndrome_list_x, syndrome_list_z):
        syndrome_x = np.array([(1 - temp) / 2 for temp in syndrome_list_x], dtype=int)
        syndrome_z = np.array([(1 - temp) / 2 for temp in syndrome_list_z], dtype=int)
        self.get_matrix()
        bp_osd_x = BpOsdDecoder(
            np.array(self.matrix_x, dtype=int),
            error_rate=0.1,
            bp_method='product_sum',
            max_iter=7,
            schedule='serial',
            osd_method='osd_cs',  # set to OSD_0 for fast solve
            osd_order=2
        )
        bp_osd_z = BpOsdDecoder(
            np.array(self.matrix_z, dtype=int),
            error_rate=0.1,
            bp_method='product_sum',
            max_iter=7,
            schedule='serial',
            osd_method='osd_cs',  # set to OSD_0 for fast solve
            osd_order=2
        )
        decoding_x = bp_osd_x.decode(syndrome_x)
        decoding_z = bp_osd_z.decode(syndrome_z)
        return np.where(decoding_x != 0)[0],np.where(decoding_z != 0)[0]


    def commute_judge(self):
        self.get_matrix()
        judge=self.matrix_x@self.matrix_z
        return np.all(judge==0)