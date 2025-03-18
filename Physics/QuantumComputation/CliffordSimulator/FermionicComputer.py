import copy
import galois
import numpy as np
from Physics.QuantumComputation.Code.QuantumCode.MajoranaOperator import MajoranaOperator
from Physics.QuantumComputation.Helper.FiniteFieldSolve import finite_field_solve
from Physics.QuantumComputation.Code.QuantumCode.FermionicCode import FermionicCode


class FermionicComputer:
    #%%  USER：构造函数
    """""
    self.number_qubit：int对象，qubits数目
    self.probability：float对象，错误发生几率
    self.stabilizers：np.array of MajoranaOperator对象，态矢的稳定子
    self.matrix：np.array of GF(2)对象，生成元校验矩阵
    self.noise_recoder：list of int对象，记录发生错误的qubits编号
    """""
    def __init__(self):
        self.number_qubit = None
        self.probability = None
        self.stabilizers = None
        self.matrix=None
        self.noise_recoder=[]
        self.noise_recoder_x=[]
        self.noise_recoder_z=[]


    #%%  USER：定义函数
    """""
    input.number_qubit：int对象，qubit数目
    input.probability：float对象，错误发生几率
    influence：修改self.number_qubit和self.probability
    """""
    def define(self, number_qubit, probability):
        ##  数据标准化
        assert isinstance(number_qubit, int)
        assert isinstance(probability, float) or isinstance(probability, int)
        assert number_qubit > 0
        assert probability >= 0
        assert probability <= 1

        ##  赋值
        self.number_qubit = number_qubit
        self.probability = probability


    #%%  USER：初始化态矢
    """""
    input.stabilizers：list of MajoranaOperator对象，初始态矢
    influence：修改self.stabilizers
    """""
    def initialize(self,stabilizers):
        self.stabilizers = stabilizers


    #%%  USER：执行相位量子逻辑门
    """""
    input.index：int对象，量子门作用的qubit的序号
    influence：将第index个qubit上的phase gate作用在self.stabilizers上面，修改self.stabilizers
    """""
    def phase(self,index_list):

        ##  输入参数是list or np.array类型
        if isinstance(index_list, list) or isinstance(index_list,np.ndarray):
            for j in range(len(index_list)):
                for i in range(len(self.stabilizers)):
                    x = np.any(self.stabilizers[i].x_vector==index_list[j])
                    z = np.any(self.stabilizers[i].z_vector==index_list[j])
                    if x^z:  # 判断是否需要变号
                        self.stabilizers[i].coff = -self.stabilizers[i].coff

        ##  输入的参数是int类型
        elif isinstance(index_list, int):
            for i in range(len(self.stabilizers)):
                x = np.any(self.stabilizers[i].x_vector==index_list)
                z = np.any(self.stabilizers[i].z_vector==index_list)
                if x^z:  # 判断是否需要变号
                    self.stabilizers[i].coff = -self.stabilizers[i].coff

        ##  补充执行噪声
        # self.dephase(index_list)


    #%%  USER：执行编织量子逻辑门
    """
    input.index_0：int对象，第一个位置
    input.index_1：int对象，第二个位置
    influence：修改self.stabilizers
    """
    def braid(self,index_0,index_1):
        assert isinstance(index_0, int) and isinstance(index_1, int)
        assert index_0 < index_1
        GF = galois.GF(2 ** 1)

        for i in range(len(self.stabilizers)):

            ##  判断依据初始化
            x_0=np.any(self.stabilizers[i].x_vector==index_0)
            x_1=np.any(self.stabilizers[i].x_vector==index_1)
            z_0=np.any(self.stabilizers[i].z_vector==index_0)
            z_1=np.any(self.stabilizers[i].z_vector==index_1)
            number_cross=0
            temp=self.stabilizers[i].get_matrix(self.number_qubit)  # 生成元矩阵化

            ##  计算跨越的算符数目
            x_vector_temp=self.stabilizers[i].x_vector
            z_vector_temp=self.stabilizers[i].z_vector
            number_cross+=len(np.where((x_vector_temp>index_0) & (x_vector_temp<index_1))[0])
            number_cross+=len(np.where((z_vector_temp>index_0) & (z_vector_temp<index_1))[0])

            ##  γ0存在的情况
            if x_0:
                number_cross_temp=number_cross+int(temp[index_0+self.number_qubit])
                number_cross_temp = number_cross_temp + int(temp[index_1])
                if np.mod(number_cross_temp,2)==0:
                    self.stabilizers[i].coff=-self.stabilizers[i].coff
                temp[index_1+self.number_qubit]+=GF(1)
                temp[index_0]+=GF(1)

            ##  γ1存在的情况
            if x_1:
                if np.mod(number_cross, 2) == 0:
                    self.stabilizers[i].coff=-self.stabilizers[i].coff
                temp[index_0+self.number_qubit]+=GF(1)
                temp[index_1]+=GF(1)

            ##  γ'0存在的情况
            if z_0:
                number_cross_temp = number_cross + int(temp[index_1])
                if np.mod(number_cross_temp, 2) == 1:
                    self.stabilizers[i].coff = -self.stabilizers[i].coff
                temp[index_1]+=GF(1)
                temp[index_0+self.number_qubit]+=GF(1)

            ##  γ'1存在的情况
            if z_1:
                number_cross_temp = number_cross + int(temp[index_0 + self.number_qubit])
                number_cross_temp = number_cross_temp + int(temp[index_1])
                if np.mod(number_cross_temp, 2) == 1:
                    self.stabilizers[i].coff=-self.stabilizers[i].coff
                temp[index_0]+=GF(1)
                temp[index_1+self.number_qubit]+=GF(1)

            ##  还原形式
            x_vector=[term for term in np.where(temp==1)[0] if term<self.number_qubit]
            z_vector=[term-self.number_qubit for term in np.where(temp==1)[0] if term>=self.number_qubit]
            self.stabilizers[i]=MajoranaOperator(x_vector,z_vector,self.stabilizers[i].coff)

        ##  补充执行噪声
        # self.loss_x([index_0,index_1])
        # self.loss_z([index_0,index_1])


    #%%  USER：执行交换量子逻辑门
    """
    input.index_0：int对象，第一个位置
    input.index_1：int对象，第二个位置
    influence：修改self.stabilizers
    """
    def fsawp(self,index_0,index_1):
        ##  输入参数是list or np.array类型
        assert isinstance(index_0, int) and isinstance(index_1, int)
        GF = galois.GF(2 ** 1)
        for i in range(len(self.stabilizers)):
            x_0=np.any(self.stabilizers[i].x_vector==index_0)
            x_1=np.any(self.stabilizers[i].x_vector==index_1)
            z_0=np.any(self.stabilizers[i].z_vector==index_0)
            z_1=np.any(self.stabilizers[i].z_vector==index_1)
            temp=GF(np.zeros(self.number_qubit*2,dtype=int))
            temp[self.stabilizers[i].x_vector]=1
            temp[self.stabilizers[i].z_vector+self.number_qubit]=1
            number_cross=0
            for j in range(index_0, index_1):
                if temp[j]==1:
                    number_cross+=1
                if temp[j+self.number_qubit]==1:
                    number_cross+=1
            if x_0:
                number_cross_temp = number_cross + int(temp[index_0 + self.number_qubit])
                if np.mod(number_cross_temp,2)==1:
                    self.stabilizers[i].coff=-self.stabilizers[i].coff
                temp[index_1]+=GF(1)
                temp[index_0]+=GF(1)
            if x_1:
                number_cross_temp = number_cross + int(temp[index_0 + self.number_qubit])
                if np.mod(number_cross_temp, 2) == 1:
                    self.stabilizers[i].coff=-self.stabilizers[i].coff
                temp[index_0]+=GF(1)
                temp[index_1]+=GF(1)
            if z_0==1:
                number_cross_temp = number_cross + int(temp[index_1])
                if np.mod(number_cross_temp, 2) == 1:
                    self.stabilizers[i].coff = -self.stabilizers[i].coff
                temp[index_1+self.number_qubit]+=GF(1)
                temp[index_0+self.number_qubit]+=GF(1)
            if z_1==1:
                number_cross_temp = number_cross + int(temp[index_1])
                if np.mod(number_cross_temp, 2) == 1:
                    self.stabilizers[i].coff=-self.stabilizers[i].coff
                temp[index_0+self.number_qubit]+=GF(1)
                temp[index_1+self.number_qubit]+=GF(1)
            x_vector=[term for term in np.where(temp==1)[0] if term<self.number_qubit]
            z_vector=[term-self.number_qubit for term in np.where(temp==1)[0] if term>=self.number_qubit]
            self.stabilizers[i]=MajoranaOperator(x_vector,z_vector,self.stabilizers[i].coff)

        ##  补充执行噪声
        # self.loss_x([index_0, index_1])
        # self.loss_z([index_0, index_1])


    #%%  USER：执行纠错
    """
    input.code：MajoranaCode对象，使用的纠错码
    input.type：str对象，纠错的方式
    """
    def correct(self, code,method):
        ##  参数标准化
        assert isinstance(code, FermionicCode)

        if method == 'common':
            ##  错误征状初始化，对应code中的checkers
            syndrome_list = np.zeros(code.number_checker, dtype=int)  # X类型错误
            for i in range(code.number_checker):
                temp=code.check_list[i].copy()
                syn,flag = self.measure(temp)
                syndrome_list[i] = int(np.real(syn))

            ##  解码
            index_list_x,index_list_z = code.decoder(syndrome_list)

            ##  探测到错误时执行纠错
            if len(index_list_x)!=0 or len(index_list_z)!=0:
                for i in self.noise_recoder_x:
                    if i in index_list_x:
                        self.noise_recoder_x.remove(i)
                for i in self.noise_recoder_z:
                    if i in index_list_z:
                        self.noise_recoder_z.remove(i)
                for i in range(len(index_list_x)):
                    self.pip_x(index_list_x[i])
                for i in range(len(index_list_z)):
                    self.pip_z(index_list_z[i])


        elif method=='relabel':
            for i in range(code.number_checker):
                temp=code.check_list[i].copy()
                syn,flag = self.measure(temp)
                if syn==-1:
                    code.check_list[i].coff=-code.check_list[i].coff


    def get_matrix(self):
        GF=galois.GF(2**1)
        self.matrix=GF(np.zeros((len(self.stabilizers),self.number_qubit*2),dtype=int))
        for i in range(len(self.stabilizers)):
            temp_x=self.stabilizers[i].x_vector
            temp_z=self.stabilizers[i].z_vector
            self.matrix[i,temp_x]=1
            self.matrix[i,temp_z+self.number_qubit]=1
        return self.matrix


    #%%  USER：测量
    def measure(self,op):
        assert isinstance(op, MajoranaOperator)
        ##  判断算符是否和stabilizers反对易
        flag=-1
        for i in range(len(self.stabilizers)):
            ##  反对易的话需要处理
            if not op.commute(self.stabilizers[i]):
                if flag==-1:
                    flag=i
                else:
                    self.stabilizers[i]=self.stabilizers[i].mul(self.stabilizers[flag],self.number_qubit)

        ##  如果反对易，那么做一次量子随机跃迁
        if flag!=-1:
            self.stabilizers[flag]=op.copy()
            if np.random.random()<0.5:
                self.stabilizers[flag].coff=-self.stabilizers[flag].coff
            return self.stabilizers[flag].coff*op.coff,flag


        ##  如果都不反对易，那么势必处于stabilizers内
        else:
            matrix = self.get_matrix()
            measurement=op.get_matrix(self.number_qubit)
            occupy_list=finite_field_solve(matrix,measurement)
            occupy_list = np.where(np.array(occupy_list) != 0)[0]
            op_multi=None
            for i in range(len(occupy_list)):
                if i==0:
                    op_multi=self.stabilizers[occupy_list[i]]
                else:
                    op_multi = op_multi.mul(self.stabilizers[occupy_list[i]],self.number_qubit)
            value=op_multi.coff*op.coff
            return value,None


    #%%  USER：发生量子错误
    """""
    input.args：int or list of int对象，发生量子错误的sites的序号
    influence：按self.probability几率在相应的sites上发生量子错误，修改self.psi
    """""
    def dephase(self, index_list):

        ##  输入参数是int的情况
        if isinstance(index_list,int) or isinstance(index_list,np.integer):
            if self.probability>np.random.rand():
                self.noise_recoder.append(index_list)
                for i in range(len(self.stabilizers)):
                    x = np.any(self.stabilizers[i].x_vector==index_list)
                    z = np.any(self.stabilizers[i].z_vector==index_list)
                    if x^z:
                        self.stabilizers[i].coff = -self.stabilizers[i].coff


        ##  输入参数是list or np.array的情况
        elif isinstance(index_list,list) or isinstance(index_list,np.ndarray) or isinstance(index_list, range):
            for i in range(len(index_list)):
                if self.probability>np.random.rand():
                    self.noise_recoder.append(index_list[i])
                    for j in range(len(self.stabilizers)):
                        x = np.any(self.stabilizers[i].x_vector == index_list[i])
                        z = np.any(self.stabilizers[i].z_vector == index_list[i])
                        if x ^ z:
                            self.stabilizers[j].coff = -self.stabilizers[j].coff


    def pip_x(self,index_list):
        if isinstance(index_list, int) or isinstance(index_list,np.integer):
            for i in range(len(self.stabilizers)):
                temp=MajoranaOperator([index_list],[],1)
                if not temp.commute(self.stabilizers[i]):
                    self.stabilizers[i].coff = -self.stabilizers[i].coff
        elif isinstance(index_list, list) or isinstance(index_list, np.ndarray) or isinstance(index_list, range):
            for i in range(len(index_list)):
                self.pip_x(index_list[i])
        else:
            raise Exception

        # self.loss_x(index_list)
        # self.loss_z(index_list)


    def pip_z(self,index_list):
        if isinstance(index_list,int) or isinstance(index_list,np.integer):
            for i in range(len(self.stabilizers)):
                temp=MajoranaOperator([],[index_list],1)
                if not temp.commute(self.stabilizers[i]):
                    self.stabilizers[i].coff = -self.stabilizers[i].coff
        elif isinstance(index_list,list) or isinstance(index_list,np.ndarray) or isinstance(index_list, range):
            for i in range(len(index_list)):
                self.pip_z(index_list[i])
        else:
            raise Exception

        # self.loss_x(index_list)
        # self.loss_z(index_list)


    def loss_x(self,index_list):
        if isinstance(index_list,int) or isinstance(index_list,np.integer):
            if self.probability>np.random.rand():
                self.noise_recoder_x.append(index_list)
                for i in range(len(self.stabilizers)):
                    temp = MajoranaOperator([index_list], [], 1)
                    if not temp.commute(self.stabilizers[i]):
                        self.stabilizers[i].coff = -self.stabilizers[i].coff
        elif isinstance(index_list,list) or isinstance(index_list,np.ndarray) or isinstance(index_list, range):
            for i in range(len(index_list)):
                self.loss_x(index_list[i])
        else:
            raise Exception


    def loss_z(self,index_list):
        if isinstance(index_list,int) or isinstance(index_list,np.integer):
            if self.probability>np.random.rand():
                self.noise_recoder_z.append(index_list)
                for i in range(len(self.stabilizers)):
                    temp = MajoranaOperator([],[index_list], 1)
                    if not temp.commute(self.stabilizers[i]):
                        self.stabilizers[i].coff = -self.stabilizers[i].coff
        elif isinstance(index_list,list) or isinstance(index_list,np.ndarray) or isinstance(index_list, range):
            for i in range(len(index_list)):
                self.loss_z(index_list[i])
        else:
            raise Exception


    #%%  USER：判断两个态矢相同
    """
    input.other：FermionicComputer对象，与本对象态矢比较
    output：bool对象，判断结果
    """
    def equal(self, other, logical_list):
        assert isinstance(other, FermionicComputer)
        stabilizer_0 = self.stabilizers
        stabilizer_1 = other.stabilizers

        if len(logical_list)==0:
            for i in range(len(stabilizer_0)):
                coff_temp_0=self.stabilizers[i].coff
                matrix = np.zeros((len(stabilizer_1),len(stabilizer_0[i])),dtype=int)
                GF=galois.GF(2**1)
                matrix=GF(matrix)
                for j in range(len(stabilizer_1)):
                    matrix[j]=stabilizer_1[j].copy()
                occupy_list = finite_field_solve(matrix,stabilizer_0[i])
                coff_temp_side=1
                for j in range(len(occupy_list)):
                    coff_temp_side=coff_temp_side*(other.stabilizers[i].coff**occupy_list [j])
                if coff_temp_0!=coff_temp_side:
                    return False
            return True
        else:
            GF = galois.GF(2 ** 1)
            stabilizer_list = []
            for i in range(2**len(logical_list)):
                for j in range(2**len(logical_list)):
                    psi_list_x=list(map(int, list(bin(i)[2:].zfill(len(logical_list)))))
                    psi_list_z=list(map(int, list(bin(j)[2:].zfill(len(logical_list)))))
                    temp=np.zeros(self.number_qubit*2,dtype=int)
                    for k in range(len(psi_list_x)):
                        if psi_list_x[k]==0:
                            pass
                        elif psi_list_x[k]==1:
                            temp[logical_list[k]]=1
                        if psi_list_z[k]==0:
                            pass
                        elif psi_list_z[k]==1:
                            temp[np.array(logical_list[k],dtype=int)+self.number_qubit]=1
                    stabilizer_list.append(temp)
            stabilizer_list.pop(0)
            matrix_0 = np.zeros((len(stabilizer_0),self.number_qubit*2), dtype=int)
            matrix_1 = np.zeros((len(stabilizer_1), self.number_qubit * 2), dtype=int)
            GF = galois.GF(2 ** 1)
            matrix_0 = GF(matrix_0)
            matrix_1 = GF(matrix_1)

            for j in range(len(stabilizer_0)):
                matrix_0[j] = stabilizer_0[j].copy()
            for j in range(len(stabilizer_1)):
                matrix_1[j] = stabilizer_1[j].copy()

            occupy_list_0=[]
            occupy_list_1 = []
            for i in range(len(stabilizer_list)):
                occupy_list_0.append(finite_field_solve(matrix_0, stabilizer_list[i]))
                occupy_list_1.append(finite_field_solve(matrix_1, stabilizer_list[i]))

            coff_list_0=[]
            coff_list_1=[]
            for i in range(len(occupy_list_0)):
                temp_0=1
                temp_1=1
                if occupy_list_0[i]==None and occupy_list_1[i]==None:
                    coff_list_0.append(None)
                    coff_list_1.append(None)
                    continue
                elif occupy_list_0[i] is None or occupy_list_1[i] is None:
                    return False

                for j in range(len(occupy_list_0[i])):
                    if occupy_list_0[i] is not None and occupy_list_1[i] is not None:
                        temp_0 = temp_0 * (self.stabilizers[j].coff ** occupy_list_0[i][j])
                        temp_1 = temp_1 * (other.stabilizers[j].coff ** occupy_list_1[i][j])
                        coff_list_0.append(temp_0)
                        coff_list_1.append(temp_1)
                    elif occupy_list_0[i] is None or occupy_list_1[i] is None:
                        return False

            for i in range(len(coff_list_0)):
                if coff_list_0[i]!=coff_list_1[i]:
                    return False
            return True

    def get_value(self,op):
        matrix = self.get_matrix()
        measurement = op.get_matrix(self.number_qubit)
        occupy_list = finite_field_solve(matrix, measurement)
        if occupy_list is None:
            return None,None
        else:
            occupy_list=np.where(np.array(occupy_list)!= 0)[0]
        op_multi = None
        for i in range(len(occupy_list)):
            if i == 0:
                op_multi = self.stabilizers[occupy_list[i]]
            else:
                op_multi = op_multi.mul(self.stabilizers[occupy_list[i]], self.number_qubit)
        return op_multi.coff,occupy_list


    #%%  USER：复制函数
    def copy(self):
        return copy.deepcopy(self)