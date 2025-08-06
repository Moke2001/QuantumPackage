import copy
import galois
import numpy as np
from ldpc import BpOsdDecoder
from MokeQuantumComputation.Code.ClassicalCode.LinearCode import LinearCode
from MokeQuantumComputation.Code.QuantumCode.Qubit.PauliOperator import PauliOperator


class PauliCode:
    #%%  USER：构造函数
    """""
    self.number_qubit：int对象，纠错码包含的qubits数目
    self.qubit_list：list of int对象，纠错码的qubits序号序列
    self.check_list：list of MajoranaOperator对象，校验子算符
    self.matrix：np.array of GF(2)对象，校验矩阵
    self.number_checker：int对象，校验子数目
    self.rank：int对象，校验矩阵的秩
    self.distance：int对象，纠错码的码距
    self.number_logical：int对象，纠错码的逻辑qubits数目
    self.logical_operator_list_x：list of MajoranaOperator对象，纠错码的逻辑γ算符
    self.logical_operator_list_z：list of MajoranaOperator对象，纠错码的逻辑γ'算符
    """""
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


    #%%  USER：添加校验子
    """
    input.number_qubit：int对象，纠错码包含的qubits数目
    influence：修改self.number_qubit和self.qubit_list
    """
    def define_qubit(self, number_qubit):
        self.number_qubit=number_qubit
        self.qubit_list=range(number_qubit)


    #%%  USER：添加校验子
    """
    input.checker：MajoranaOperator对象，校验子
    input.args：str对象，校验子类型
    influence：在self.check_list中添加校验子
    """
    def push(self,checker,*args):
        if len(args)==0:
            assert isinstance(checker,PauliOperator)
            self.check_list.append(checker)
            self.number_checker=self.number_checker+1
        else:
            if args[0]=='x' or args[0]=='X':
                self.check_list.append(PauliOperator(checker,[],1))
            elif args[0]=='z' or args[0]=='Z':
                self.check_list.append(PauliOperator([],checker,1))
            self.number_checker=self.number_checker+1


    #%%  USER：添加qubit
    """
    input.number：int对象，添加的数目
    influence：修改self.number_qubit和self.qubit_list
    """
    def push_qubit(self,number):
        assert isinstance(number,int)
        self.number_qubit=self.number_qubit+number
        self.qubit_list=range(self.number_qubit)


    #%%  USER：用校验矩阵定义纠错码
    """
    input.matrix：np.array of GF(2)对象，校验矩阵
    influence：对self.matrix，self.qubit_list，self.number_qubit和self.check_list赋值
    """
    def define_matrix(self,matrix):
        GF = galois.GF(2 ** 1)
        self.matrix = GF(np.array(matrix,dtype=int))
        self.define_qubit(self.matrix.shape[1] // 2)
        for i in range(len(self.matrix)):
            loc_x=np.where(self.matrix[i][0:self.number_qubit]==1)[0]
            loc_z=np.where(self.matrix[i][self.number_qubit:]==1)[0]
            self.push(PauliOperator(loc_x,loc_z,1))


    #%%  USER：计算校验子形成的校验矩阵
    """
    output：np.array of GF(2)对象，校验矩阵
    """
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
    def linear_combine(self,code_x,code_z):
        assert isinstance(code_x,LinearCode)
        assert isinstance(code_z,LinearCode)

        GF=galois.GF(2**1)
        matrix=GF(np.zeros((code_x.number_checker+code_z.number_checker,code_x.number_bit*2),dtype=int))
        for i in range(code_x.number_checker):
            matrix[i][0::2]=code_x.check_matrix[i]
        for i in range(code_z.number_checker):
            matrix[i][1::2]=code_z.check_matrix[i]
        self.define_matrix(matrix)


    #%%  KEY：判断校验子之间是否对易
    def commute_judge(self):
        for i in range(len(self.check_list)):
            for j in range(i+1,len(self.check_list)):
                if not self.check_list[i].commute(self.check_list[j]):
                    return False
        return True


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
                self.logical_operator_list_x.append(PauliOperator(x_vec,z_vec,1))
                self.logical_operator_list_z.append(PauliOperator(z_vec,x_vec,1))

            return self.logical_operator_list_x,self.logical_operator_list_z


    #%%  USER：基于BP+OSD算法的解码器
    """
    input.syndrome_list：list of GF(2)对象，错误征状
    output：tuple of np.array of int对象，需要处理的qubits的序号
    """
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
    """
    output：tuple of list of MajoranaOperator对象，CSS codes的校验子分类
    """
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


    #%%  USER：复制函数
    """
    output：FermionicCode对象，复制结果
    """
    def copy(self):
        return copy.deepcopy(self)


    #%%  USER：重编码
    """
    input.number_qubit：int对象，目标qubits数目
    input.mapping：list of int or np.array of int对象，重编码映射
    """
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
