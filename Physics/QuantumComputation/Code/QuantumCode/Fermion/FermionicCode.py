import copy
import itertools
import galois
import ldpc
import numpy as np
from ldpc import BpOsdDecoder

from Physics.QuantumComputation.Code.ClassicalCode.BicycleCode import BicycleCode
from Physics.QuantumComputation.Code.ClassicalCode.LinearCode import LinearCode
from Physics.QuantumComputation.Code.QuantumCode.Fermion.MajoranaOperator import MajoranaOperator
from Physics.QuantumComputation.Helper.FiniteFieldSolve import FiniteFieldSolve
from Physics.QuantumComputation.Helper.Orthogonalize import orthogonalize
from Physics.QuantumComputation.Helper.QuantumCodeDistance import QuantumCodeDistance


class FermionicCode:
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
        self.gauge_list=[]


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
            assert isinstance(checker,MajoranaOperator)
            self.check_list.append(checker)
            self.number_checker=self.number_checker+1
        else:
            if args[0]=='x' or args[0]=='X':
                self.check_list.append(MajoranaOperator(checker,[],1))
            elif args[0]=='z' or args[0]=='Z':
                self.check_list.append(MajoranaOperator([],checker,1))
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
            loc_x=np.where(self.matrix[i][0::2]==1)[0]
            loc_z=np.where(self.matrix[i][1::2]==1)[0]
            self.push(MajoranaOperator(loc_x,loc_z,1))


    #%%  USER：添加gauge qubit
    """
    input.number：list of int对象，添加gauge index的位置
    influence：修改self.gauge_list
    """
    def push_gauge(self,index_list):
        self.gauge_list.append(index_list)


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
            self.matrix[i,temp_x*2]=1
            self.matrix[i,temp_z*2+1]=1
        return self.matrix


    #%%  USER：计算校验子形成的校验矩阵
    """
    output：np.array of GF(2)对象，单边校验矩阵
    """
    def get_core_matrix(self):
        GF=galois.GF(2**1)
        matrix=GF(np.zeros((self.number_checker,self.number_qubit),dtype=int))
        for i in range(len(self.check_list)):
            temp_x=self.check_list[i].x_vector
            matrix[i,temp_x]=1
        return matrix


    #%%  USER：合并线性码生成Majorana code
    """
    input.code：LinearCode对象，dual-containing code
    influence：定义self
    """
    def linear_combine(self,code):
        assert isinstance(code,LinearCode)
        GF=galois.GF(2**1)
        matrix=GF(np.zeros((code.number_checker*2,code.number_bit*2),dtype=int))
        for i in range(code.number_checker):
            matrix[i][0::2]=code.check_matrix[i]
            matrix[i+code.number_checker][1::2] = code.check_matrix[i]
        self.define_matrix(matrix)


    #%%  KEY：判断校验子之间是否对易
    """
    output：bool对象，判断结果
    """
    def commute_judge(self):
        return np.all(self.get_matrix()@self.get_matrix().T==0)


    #%%  KEY：输出codespace
    """
    influence：输出codespace的txt
    """
    def get_codespace(self):
        matrix=self.get_core_matrix()
        GF = galois.GF(2 ** 1)
        independent_null_basis_list=[]
        codewords = matrix.null_space()
        for vec in codewords:
            rank_before = np.linalg.matrix_rank(matrix)
            matrix = np.vstack([matrix, GF(np.array(vec, dtype=int))])
            if np.linalg.matrix_rank(matrix) == rank_before + 1:
                independent_null_basis_list.append(vec.tolist())
        with open("codespace.txt", "w") as f:
            f.write("matrix="+str(independent_null_basis_list))


    #%%  USER：求逻辑算符组
    """
    output：tuple of list of MajoranaOperator对象，逻辑算符
    influence：修改self.logical_operator_list_x和self.logical_operator_list_z
    """
    def get_logical_operators(self,*args):
        if len(self.logical_operator_list_x)!=0:
            return self.logical_operator_list_x,self.logical_operator_list_z
        else:
            matrix=self.get_core_matrix()
            if len(args)==1:
                for vec in args:
                    matrix=np.vstack([matrix,vec])
            codewords = matrix.null_space()
            GF = galois.GF(2 ** 1)

            # 筛选与行空间无关的基矢
            independent_null_basis_list = []
            for vec in codewords:
                rank_before = np.linalg.matrix_rank(matrix)
                matrix = np.vstack([matrix, GF(np.array(vec, dtype=int))])
                if np.linalg.matrix_rank(matrix) == rank_before + 1:
                    independent_null_basis_list.append(vec)
            basis_list=orthogonalize(independent_null_basis_list)

            print(basis_list)
            ##  加入逻辑算符中
            for vec in basis_list:
                loc=np.where(vec==1)[0]
                if np.mod(len(loc),2)==1:
                    self.logical_operator_list_x.append(MajoranaOperator(loc,[],1))
                    self.logical_operator_list_z.append(MajoranaOperator([],loc,1))


            return self.logical_operator_list_x,self.logical_operator_list_z


    #%%  USER：求不相交逻辑算符组
    """
    output：tuple of list of MajoranaOperator对象，逻辑算符
    influence：修改self.logical_operator_list_x和self.logical_operator_list_z
    """
    def get_minimal_logical_operator(self,weight,number):
        matrix=self.get_core_matrix()
        GF=galois.GF(2 ** 1)
        pos = range(matrix.shape[1])
        result_list=[]
        for num in range(weight, matrix.shape[1]):
            for each in itertools.combinations(pos, weight):
                print(len(result_list))
                temp = GF(np.zeros(matrix.shape[1], dtype=int))
                temp[list(each)] = 1
                if np.count_nonzero(matrix @ temp) == 0:
                    if FiniteFieldSolve(matrix, temp) is None:
                        flag=True
                        for vec in result_list:
                            if np.dot(np.array(vec),np.array(temp))>0:
                                flag=False
                                break
                        if flag:
                            result_list.append(temp)
                if len(result_list)==number:
                    break
            if len(result_list)==number:
                break
        for i in range(len(result_list)):
            temp=np.where(result_list[i]==1)[0]
            temp_x=MajoranaOperator(temp,[],1)
            temp_z=MajoranaOperator([],temp,1)
            self.logical_operator_list_x.append(temp_x)
            self.logical_operator_list_z.append(temp_z)


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
        pip_x_list = decoding[0::2]
        pip_z_list = decoding[1::2]
        return np.where(pip_x_list!=0)[0],np.where(pip_z_list!=0)[0]


    #%%  USER：CSS码专用
    """
    output：tuple of list of MajoranaOperator对象，CSS codes的校验子分类
    """
    def css(self):
        for i in range(len(self.check_list)):
            if len(self.check_list[i].x_vector)>0 and len(self.check_list[i].z_vector)>0:
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


    #%%  USER：返回码距
    """
    output：int对象，码距
    """
    def get_distance(self):
        gauge_list=[]
        GF=galois.GF(2)
        for i in range(len(self.gauge_list)):
            temp=GF(np.zeros(self.number_qubit,dtype=int))
            temp[self.gauge_list[i]]=1
            gauge_list.append(temp)
        return QuantumCodeDistance(self.get_core_matrix(), gauge_list)


if __name__ == '__main__':
    for i in range(1000):
        code_linear=BicycleCode(24,10,10,i)
        dis=code_linear.get_distance()
        print(dis)
        if dis>=4:
            print(i)
    # code=FermionicCode()
    # code.linear_combine(code_linear)
    # print(code.get_distance())