import copy

import galois
import numpy as np
from ldpc import BpOsdDecoder

from Physics.QuantumComputation.Code.ClassicalCode.EuclideanCode import EuclideanCode
from Physics.QuantumComputation.Code.ClassicalCode.LinearCode import LinearCode
from Physics.QuantumComputation.Code.QuantumCode.MajoranaOperator import MajoranaOperator


class FermionicCode:
    #%%  USER：构造函数
    def __init__(self):
        self.number_qubit=0
        self.qubit_list=[]
        self.check_list=[]
        self.matrix=None
        self.number_checker=0
        self.rank=None
        self.distance=None
        self.number_logical=None
        self.logical_operator_list_x=[]
        self.logical_operator_list_z=[]


    def define_qubit(self, number_qubit):
        self.number_qubit=number_qubit
        self.qubit_list=range(number_qubit)


    def push(self,checker,*args):
        if len(args)==0:
            assert isinstance(checker,MajoranaOperator)
            self.check_list.append(checker)
            self.number_checker=self.number_checker+1
        else:
            if args[0]=='x' or args[0]=='X':
                self.check_list.append(MajoranaOperator(checker,[],1))
            elif args[0]=='z' or args[0]=='Z':
                self.check_list.append(MajoranaOperator([],checker,1))
            self.number_checker=self.number_checker+1

    def push_qubit(self,number):
        assert isinstance(number,int)
        self.number_qubit=self.number_qubit+number
        self.qubit_list=range(self.number_qubit)

    def define_matrix(self,matrix):
        GF = galois.GF(2 ** 1)
        self.matrix = GF(np.array(matrix,dtype=int))
        self.define_qubit(self.matrix.shape[1] // 2)
        for i in range(len(self.matrix)):
            loc_x=np.where(self.matrix[i][0:self.number_qubit]==1)[0]
            loc_z=np.where(self.matrix[i][self.number_qubit:]==1)[0]
            self.push(MajoranaOperator(loc_x,loc_z,1))


    def get_matrix(self):
        GF=galois.GF(2**1)
        self.matrix=GF(np.zeros((self.number_checker,self.number_qubit*2),dtype=int))
        for i in range(len(self.check_list)):
            temp_x=self.check_list[i].x_vector
            temp_z=self.check_list[i].z_vector
            self.matrix[i,temp_x]=1
            self.matrix[i,temp_z+self.number_qubit]=1
        return self.matrix


    #%%  USER：合并线性码生成Majorana code
    def linear_combine(self,code):
        assert isinstance(code,LinearCode)
        GF=galois.GF(2**1)
        matrix=GF(np.zeros((code.number_checker*2,code.number_bit*2),dtype=int))
        for i in range(code.number_checker):
            matrix[i][0:code.number_bit]=code.check_matrix[i]
            matrix[i+code.number_checker][code.number_bit:] = code.check_matrix[i]
        self.define_matrix(matrix)


    #%%  KEY：判断校验子之间是否对易
    def commute_judge(self):
        return np.all(self.get_matrix()@self.get_matrix().T==0)


    #%%  KEY：输出codespace
    def get_codespace(self):
        self.get_matrix()
        codewords = self.get_matrix().null_space()
        with open("codespace.txt", "w") as f:
            f.write("matrix="+str(codewords.tolist()))


    #%%  USER：求逻辑算符组
    def get_logical_operators(self):
        if len(self.logical_operator_list_x)!=0:
            return self.logical_operator_list_x,self.logical_operator_list_z
        else:
            self.get_matrix()
            codewords = self.matrix.null_space()
            GF = galois.GF(2 ** 1)
            matrix = self.matrix.copy()

            # 筛选与行空间无关的基矢
            independent_null_basis_list = []
            for vec in codewords:
                rank_before = np.linalg.matrix_rank(matrix)
                matrix = np.vstack([matrix, GF(np.array(vec, dtype=int))])
                if np.linalg.matrix_rank(matrix) == rank_before + 1:
                    independent_null_basis_list.append(vec)
                else:
                    matrix = np.delete(matrix, -1, axis=0)
            independent_null_basis_list = []

            # 正交化过程X
            ortho_basis_x = []
            for vec in independent_null_basis_list:
                v = vec.copy()  # 复制向量以避免修改原始数据

                ##  对现有正交基矢进行正交化
                for u in ortho_basis_x:
                    dot_product = np.dot(v, u)
                    if dot_product:
                        v += u
                ortho_basis_x.append(v)

            ##  加入逻辑算符中
            for vec in ortho_basis_x:
                x_vec=[temp for temp in np.where(vec==1)[0] if temp<self.number_qubit]
                z_vec=[temp for temp in np.where(vec==1)[0] if temp>=self.number_qubit]
                self.logical_operator_list_x.append(MajoranaOperator(x_vec,z_vec,1))
                self.logical_operator_list_z.append(MajoranaOperator(z_vec,x_vec,1))

            return self.logical_operator_list_x,self.logical_operator_list_z

    #%%  USER：基于BP+OSD算法的解码器
    def decoder(self,syndrome_list):
        syndrome = np.array([(1-temp)/2 for temp in syndrome_list],dtype=int)
        self.get_matrix()
        bp_osd = BpOsdDecoder(
            np.array(self.get_matrix(),dtype=int),
            error_rate=0.1,
            bp_method='product_sum',
            max_iter=7,
            schedule='serial',
            osd_method='osd_cs',  # set to OSD_0 for fast solve
            osd_order=2
        )
        decoding = bp_osd.decode(syndrome)
        pip_x_list = decoding[0:self.number_qubit]
        pip_z_list = decoding[self.number_qubit:]
        return np.where(pip_x_list!=0)[0],np.where(pip_z_list!=0)[0]


    #%%  USER：CSS码专用
    def css(self):
        n=self.number_qubit
        for i in range(len(self.check_list)):
            if len(self.check_list[i].x_vector)>0 and len(self.check_list[i].z_vector)>0 :
                return False
        check_list_x = []
        check_list_z = []
        for i in range(len(self.check_list)):
            if len(self.check_list[i].x_vector)>0:
                check_list_x.append(self.check_list[i])
            else:
                check_list_z.append(self.check_list[i])
        return check_list_x, check_list_z


    def copy(self):
        return copy.deepcopy(self)


    def index_map(self,number_qubit,mapping):
        self.define_qubit(number_qubit)
        for i in range(len(self.check_list)):
            for j in range(len(self.check_list[i].x_vector)):
                self.check_list[i].x_vector[j]=mapping[self.check_list[i].x_vector[j]]
            for j in range(len(self.check_list[i].z_vector)):
                self.check_list[i].z_vector[j]=mapping[self.check_list[i].z_vector[j]]
        for i in range(len(self.logical_operator_list_x)):
            for j in range(len(self.logical_operator_list_x[i].x_vector)):
                self.logical_operator_list_x[i].x_vector[j]=mapping[self.logical_operator_list_x[i].x_vector[j]]
            for j in range(len(self.logical_operator_list_x[i].z_vector)):
                self.logical_operator_list_x[i].z_vector[j]=mapping[self.logical_operator_list_x[i].z_vector[j]]
            for j in range(len(self.logical_operator_list_z[i].x_vector)):
                self.logical_operator_list_z[i].x_vector[j]=mapping[self.logical_operator_list_z[i].x_vector[j]]
            for j in range(len(self.logical_operator_list_z[i].z_vector)):
                self.logical_operator_list_z[i].z_vector[j]=mapping[self.logical_operator_list_z[i].z_vector[j]]

if __name__=='__main__':
    code=FermionicCode()
    linear_code=EuclideanCode(2,2,2)
    code.linear_combine(linear_code)
    print(code.get_matrix())