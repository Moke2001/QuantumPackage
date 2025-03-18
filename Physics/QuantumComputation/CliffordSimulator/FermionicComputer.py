import copy
import galois
import numpy as np
from Physics.QuantumComputation.Code.QuantumCode.Fermion.MajoranaOperator import MajoranaOperator
from Physics.QuantumComputation.Helper.FiniteFieldSolve import FiniteFieldSolve
from Physics.QuantumComputation.Code.QuantumCode.Fermion.FermionicCode import FermionicCode


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
        stabilizers=stabilizers.copy()
        self.stabilizers = np.array(stabilizers,dtype=MajoranaOperator)


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
        self.dephase(index_list)


    #%%  USER：执行编织量子逻辑门
    """
    input.index_0：int对象，第一个位置
    input.index_1：int对象，第二个位置
    influence：修改self.stabilizers
    """
    def braid(self,index_0,index_1):
        assert isinstance(index_0, int) and isinstance(index_1, int)
        assert index_0 < index_1

        for i in range(len(self.stabilizers)):

            ##  判断依据初始化
            x_0=np.where(self.stabilizers[i].x_vector==index_0)[0]
            x_1=np.where(self.stabilizers[i].x_vector==index_1)[0]
            z_0=np.where(self.stabilizers[i].z_vector==index_0)[0]
            z_1=np.where(self.stabilizers[i].z_vector==index_1)[0]
            number_cross=0

            ##  计算跨越的算符数目
            x_vector_temp=self.stabilizers[i].x_vector.copy()
            z_vector_temp=self.stabilizers[i].z_vector.copy()
            number_cross+=len(np.where((x_vector_temp>index_0) & (x_vector_temp<index_1))[0])
            number_cross+=len(np.where((z_vector_temp>index_0) & (z_vector_temp<index_1))[0])
            number_exist=len(x_0)+len(x_1)+len(z_0)+len(z_1)
            number_cross=number_cross*number_exist
            if number_exist==0:
                continue
            del_list_x=[]
            app_list_x=[]
            del_list_z=[]
            app_list_z=[]
            if len(x_0)==1 and len(z_1)==1:
                if len(z_0)==1:
                    number_cross+=2
                if len(x_1)==1:
                    number_cross+=2
                number_cross+=2
            elif len(x_0)==1 and len(z_1)==0:
                if len(z_0)==1:
                    number_cross+=1
                if len(x_1)==1:
                    number_cross+=1
                number_cross+=1
                del_list_x.append(x_0[0])
                app_list_z.append(index_1)
            elif len(x_0)==0 and len(z_1)==1:
                if len(z_0)==1:
                    number_cross+=1
                if len(x_1)==1:
                    number_cross+=1
                del_list_z.append(z_1[0])
                app_list_x.append(index_0)
            elif len(x_0)==0 and len(z_1)==0:
                pass
            else:
                raise NotImplementedError

            if len(z_0)==1 and len(x_1)==1:
                number_cross+=2
            elif len(z_0)==1 and len(x_1)==0:
                del_list_z.append(z_0[0])
                app_list_x.append(index_1)
            elif len(z_0)==0 and len(x_1)==1:
                number_cross += 1
                del_list_x.append(x_1[0])
                app_list_z.append(index_0)
            elif len(x_1)==0 and len(z_0)==0:
                pass
            else:
                raise NotImplementedError

            if np.mod(number_cross, 2) == 1:
                self.stabilizers[i].coff = -self.stabilizers[i].coff

            x_vector_temp=np.delete(x_vector_temp,np.array(del_list_x,dtype=int))
            x_vector_temp=np.append(x_vector_temp,np.array(app_list_x,dtype=int))
            z_vector_temp=np.delete(z_vector_temp,np.array(del_list_z,dtype=int))
            z_vector_temp=np.append(z_vector_temp,np.array(app_list_z,dtype=int))

            self.stabilizers[i]=MajoranaOperator(x_vector_temp,z_vector_temp,self.stabilizers[i].coff)

        ##  补充执行噪声
        #self.depolarize([index_0,index_1])


    #%%  USER：执行交换量子逻辑门
    """
    input.index_0：int对象，第一个位置
    input.index_1：int对象，第二个位置
    influence：修改self.stabilizers
    """
    def fsawp(self,index_0,index_1):
        ##  输入参数是list or np.array类型
        assert isinstance(index_0, int) and isinstance(index_1, int)
        for i in range(len(self.stabilizers)):
            x_0=np.where(self.stabilizers[i].x_vector==index_0)[0]
            x_1=np.where(self.stabilizers[i].x_vector==index_1)[0]
            z_0=np.where(self.stabilizers[i].z_vector==index_0)[0]
            z_1=np.where(self.stabilizers[i].z_vector==index_1)[0]
            number_cross=0

            ##  计算跨越的算符数目
            x_vector_temp=self.stabilizers[i].x_vector.copy()
            z_vector_temp=self.stabilizers[i].z_vector.copy()
            number_cross+=len(np.where((x_vector_temp>index_0) & (x_vector_temp<index_1))[0])
            number_cross+=len(np.where((z_vector_temp>index_0) & (z_vector_temp<index_1))[0])
            number_exist=len(x_0)+len(x_1)+len(z_0)+len(z_1)
            number_cross=number_cross*number_exist
            if number_exist==0:
                continue

            del_list_x = []
            app_list_x = []
            del_list_z = []
            app_list_z = []
            if len(x_0) == 1 and len(x_1) == 1:
                if len(z_0) == 1:
                    number_cross += 2
                number_cross += 1
            elif len(x_0) == 1 and len(x_1) == 0:
                if len(z_0) == 1:
                    number_cross += 1
                del_list_x.append(x_0[0])
                app_list_x.append(index_1)
            elif len(x_0) == 0 and len(x_1) == 1:
                if len(z_0) == 1:
                    number_cross += 1
                del_list_x.append(x_1[0])
                app_list_x.append(index_0)
            elif len(x_0) == 0 and len(x_1) == 0:
                pass
            else:
                raise NotImplementedError

            if len(z_0) == 1 and len(z_1) == 1:
                if len(x_0) == 1:
                    number_cross += 2
                number_cross += 1
            elif len(z_0) == 1 and len(z_1) == 0:
                if len(x_0) == 1:
                    number_cross += 1
                del_list_z.append(z_0[0])
                app_list_z.append(index_1)
            elif len(z_0) == 0 and len(z_1) == 1:
                if len(x_0) == 1:
                    number_cross += 1
                del_list_z.append(z_1[0])
                app_list_z.append(index_0)
            elif len(z_0) == 0 and len(z_1) == 0:
                pass
            else:
                raise NotImplementedError

            if np.mod(number_cross, 2) == 1:
                self.stabilizers[i].coff = -self.stabilizers[i].coff

            x_vector_temp = np.delete(x_vector_temp, np.array(del_list_x, dtype=int))
            x_vector_temp = np.append(x_vector_temp, np.array(app_list_x, dtype=int))
            z_vector_temp = np.delete(z_vector_temp, np.array(del_list_z, dtype=int))
            z_vector_temp = np.append(z_vector_temp, np.array(app_list_z, dtype=int))

            self.stabilizers[i] = MajoranaOperator(x_vector_temp, z_vector_temp, self.stabilizers[i].coff)

        ##  补充执行噪声
        #self.depolarize([index_0, index_1])


    #%%  USER：执行纠错
    """
    input.code：MajoranaCode对象，使用的纠错码
    input.type：str对象，纠错的方式
    """
    def correct(self, code,method,*args):

        ##  参数标准化
        assert isinstance(code, FermionicCode)

        ##  一般的纠错流程
        if len(args) == 0:
            if method == 'common':
                ##  错误征状初始化，对应code中的checkers
                syndrome_list = np.zeros(code.number_checker, dtype=int)  # X类型错误
                for i in range(code.number_checker):
                    temp=code.check_list[i].copy()
                    syn,flag = self.measure(temp)
                    syndrome_list[i] = int(np.real(syn))

                ##  解码
                index_list_x,index_list_z = code.decoder(syndrome_list)
                if len(index_list_x)!=0 or len(index_list_z)!=0:
                    temp=self.noise_recoder_x.copy()
                    for i in temp:
                        if i in index_list_x:
                            self.noise_recoder_x.remove(i)
                    temp=self.noise_recoder_z.copy()
                    for i in temp:
                        if i in index_list_z:
                            self.noise_recoder_z.remove(i)
                    self.pip_x(index_list_x)
                    self.pip_z(index_list_z)

            elif method=='relabel':
                for i in range(code.number_checker):
                    temp=code.check_list[i].copy()
                    syn,flag = self.measure(temp)
                    if syn==-1:
                        code.check_list[i].coff=-code.check_list[i].coff

        ##  快速模拟
        else:
            stabilizer_index_vector=np.array(args[0],dtype=int)
            if method == 'common':

                ##  错误征状初始化，对应code中的checkers
                syndrome_list = np.zeros(code.number_checker, dtype=int)  # X类型错误
                for i in range(code.number_checker):
                    if stabilizer_index_vector[i] is not None:
                        syn=self.stabilizers[stabilizer_index_vector[i]].coff*code.check_list[i].coff
                    else:
                        temp = code.check_list[i].copy()
                        syn, flag = self.measure(temp)
                    syndrome_list[i] = int(np.real(syn))

                ##  解码
                index_list_x,index_list_z = code.decoder(syndrome_list)
                if len(index_list_x)!=0 or len(index_list_z)!=0:
                    temp=self.noise_recoder_x.copy()
                    for i in temp:
                        if i in index_list_x:
                            self.noise_recoder_x.remove(i)
                    temp=self.noise_recoder_z.copy()
                    for i in temp:
                        if i in index_list_z:
                            self.noise_recoder_z.remove(i)
                    self.pip_x(index_list_x)
                    self.pip_z(index_list_z)

            else:
                raise NotImplementedError


    #%%  KEY：求稳定子的校验矩阵
    def get_matrix(self):
        GF=galois.GF(2**1)
        self.matrix=GF(np.zeros((len(self.stabilizers),self.number_qubit*2),dtype=int))
        for i in range(len(self.stabilizers)):
            temp_x=self.stabilizers[i].x_vector
            temp_z=self.stabilizers[i].z_vector
            self.matrix[i,temp_x*2]=1
            self.matrix[i,temp_z*2+1]=1
        return self.matrix


    #%%  USER：测量操作
    """
    input.op：MajoranaOperator对象，待测量的算符
    output：tuple of int对象，测量值和新稳定子的位置
    influence：改变self.stabilizers
    """
    def measure(self, op):

        ##  数据标准化
        assert isinstance(op, MajoranaOperator)

        ##  判断算符是否和稳定子反对易
        flag=-1
        for i in range(len(self.stabilizers)):
            ##  记录第一个反对易稳定子的位置，其余反对易稳定子乘上第一个反对易稳定子
            if not op.commute(self.stabilizers[i]):
                if flag==-1:
                    flag=i
                else:
                    self.stabilizers[i]=self.stabilizers[i].mul(self.stabilizers[flag],self.number_qubit)

        ##  如果存在反对易算符，那么做一次量子随机跃迁
        if flag!=-1:
            self.stabilizers[flag]=op.copy()
            if np.random.random()<0.5:
                self.stabilizers[flag].coff=-self.stabilizers[flag].coff
            return self.stabilizers[flag].coff*op.coff,flag


        ##  如果都不反对易，那么势必处于stabilizers内
        else:

            ##  计算系数
            matrix = self.get_matrix()
            measurement=op.get_matrix(self.number_qubit)
            occupy_list=FiniteFieldSolve(matrix, measurement)
            occupy_list = np.where(np.array(occupy_list) != 0)[0]

            ##  求相乘结果
            op_multi=None
            for i in range(len(occupy_list)):
                if i==0:
                    op_multi=self.stabilizers[occupy_list[i]]
                else:
                    assert op_multi.commute(self.stabilizers[occupy_list[i]])
                    op_multi = op_multi.mul(self.stabilizers[occupy_list[i]],self.number_qubit)
            value=op_multi.coff*op.coff

            ##  返回结果
            return value,None


    #%%  USER：测量操作
    """
    input.op：MajoranaOperator对象，待测量的算符
    output：tuple of int对象，测量值和新稳定子的位置
    influence：改变self.stabilizers
    """
    def measure_noisy(self,op):

        ##  数据标准化
        assert isinstance(op, MajoranaOperator)

        value, flag=self.measure(op)
        if np.random.random()<self.probability:
            return -value,flag
        else:
            return value,flag


    #%%  USER：发生量子错误
    """""
    input.index_list：int or list of int对象，发生量子错误的sites的序号
    influence：按self.probability几率在相应的qubits上发生量子错误，修改self.stabilizers
    """""
    def dephase(self, index_list):

        ##  输入参数是int的情况
        if isinstance(index_list,int) or isinstance(index_list,np.integer):
            if self.probability>np.random.rand():
                self.noise_recoder_x.append(index_list)
                self.noise_recoder_z.append(index_list)
                for i in range(len(self.stabilizers)):
                    x = np.any(self.stabilizers[i].x_vector==index_list)
                    z = np.any(self.stabilizers[i].z_vector==index_list)
                    if x^z:
                        self.stabilizers[i].coff = -self.stabilizers[i].coff


        ##  输入参数是list or np.array的情况
        elif isinstance(index_list,list) or isinstance(index_list,np.ndarray) or isinstance(index_list, range):
            for i in range(len(index_list)):
                self.dephase(index_list[i])

        else:
            raise Exception


    #%%  USER：发生去极化信道量子错误
    """""
    input.index_list：int or list of int对象，发生量子错误的sites的序号
    influence：按self.probability几率在相应的qubits上发生去极化信道，修改self.stabilizers
    """""
    def depolarize(self,index_list):
        ##  输入参数是int的情况
        if isinstance(index_list,int) or isinstance(index_list,np.integer):
            if self.probability>np.random.rand():
                r = np.random.random()
                if r < 0.3333333333333:
                    self.noise_recoder_x.append(index_list)
                    temp = MajoranaOperator([index_list], [], 1)
                    for i in range(len(self.stabilizers)):
                        if not temp.commute(self.stabilizers[i]):
                            self.stabilizers[i].coff = -self.stabilizers[i].coff
                elif 0.333333333<r<0.66666666666:
                    self.noise_recoder_z.append(index_list)
                    temp = MajoranaOperator([], [index_list], 1)
                    for i in range(len(self.stabilizers)):
                        if not temp.commute(self.stabilizers[i]):
                            self.stabilizers[i].coff = -self.stabilizers[i].coff
                else:
                    self.noise_recoder_x.append(index_list)
                    self.noise_recoder_z.append(index_list)
                    temp = MajoranaOperator([index_list], [index_list], 1)
                    for i in range(len(self.stabilizers)):
                        if not temp.commute(self.stabilizers[i]):
                            self.stabilizers[i].coff = -self.stabilizers[i].coff


        ##  输入参数是list or np.array的情况
        elif isinstance(index_list,list) or isinstance(index_list,np.ndarray) or isinstance(index_list, range):
            for i in range(len(index_list)):
                self.depolarize(index_list[i])

        else:
            raise Exception


    #%%  USER：作用γ算符
    """""
    input.index_list：int or list of int对象，作用的qubits的序号
    influence：在相应的qubits上施加操作，修改self.stabilizers
    """""
    def pip_x(self,index_list):

        ##  输入参数是int的情况
        if isinstance(index_list, int) or isinstance(index_list,np.integer):
            for i in range(len(self.stabilizers)):
                temp=MajoranaOperator([index_list],[],1)
                if not temp.commute(self.stabilizers[i]):
                    self.stabilizers[i].coff = -self.stabilizers[i].coff
            #self.loss(index_list)

        ##  输入参数是list or np.array的情况
        elif isinstance(index_list, list) or isinstance(index_list, np.ndarray) or isinstance(index_list, range):
            for i in range(len(index_list)):
                self.pip_x(index_list[i])
        else:
            raise Exception


    #%%  USER：作用γ'算符
    """""
    input.index_list：int or list of int对象，作用的qubits的序号
    influence：在相应的qubits上施加操作，修改self.stabilizers
    """""
    def pip_z(self,index_list):

        ##  输入参数是int的情况
        if isinstance(index_list,int) or isinstance(index_list,np.integer):
            for i in range(len(self.stabilizers)):
                temp=MajoranaOperator([],[index_list],1)
                if not temp.commute(self.stabilizers[i]):
                    self.stabilizers[i].coff = -self.stabilizers[i].coff
            ##  补充执行噪声
            #self.loss(index_list)

        ##  输入参数是list or np.array的情况
        elif isinstance(index_list,list) or isinstance(index_list,np.ndarray) or isinstance(index_list, range):
            for i in range(len(index_list)):
                self.pip_z(index_list[i])
        else:
            raise Exception


    #%%  USER：发生原子丢失量子错误
    """""
    input.index_list：int or list of int对象，发生量子错误的sites的序号
    influence：按self.probability几率在相应的qubits上发生原子丢失量子错误，修改self.stabilizers
    """""
    def loss(self,index_list):

        ##  输入参数是int的情况
        if isinstance(index_list,int) or isinstance(index_list,np.integer):
            if self.probability>np.random.rand():
                r=np.random.random()
                if r<0.5:
                    self.noise_recoder_x.append(index_list)
                    temp = MajoranaOperator([index_list], [], 1)
                    for i in range(len(self.stabilizers)):
                        if not temp.commute(self.stabilizers[i]):
                            self.stabilizers[i].coff = -self.stabilizers[i].coff
                else:
                    self.noise_recoder_z.append(index_list)
                    temp = MajoranaOperator([], [index_list], 1)
                    for i in range(len(self.stabilizers)):
                        if not temp.commute(self.stabilizers[i]):
                            self.stabilizers[i].coff = -self.stabilizers[i].coff

        ##  输入参数是list or np.array的情况
        elif isinstance(index_list,list) or isinstance(index_list,np.ndarray) or isinstance(index_list, range):
            for i in range(len(index_list)):
                self.loss(index_list[i])

        ##  报错
        else:
            raise Exception


    #%%  USER：获取某个与稳定子对易的可观测量的值
    """
    input.op：MajoranaOperator对象，可观测量
    output：int or None对象，如果对易则给出结果，不对易给出None
    """
    def get_value(self,op):

        ##  计算是否能由稳定子生成
        matrix = self.get_matrix()
        measurement = op.get_matrix(self.number_qubit)
        occupy_list = FiniteFieldSolve(matrix, measurement)
        if occupy_list is None:
            return None,None
        else:
            occupy_list=np.where(np.array(occupy_list)!= 0)[0]

        ##  如果可以，计算结果
        op_multi = None
        for i in range(len(occupy_list)):
            if i == 0:
                op_multi = self.stabilizers[occupy_list[i]]
            else:
                op_multi = op_multi.mul(self.stabilizers[occupy_list[i]], self.number_qubit)
        return op_multi.coff*op.coff,occupy_list


    #%%  USER：复制函数
    """
    output：FermionicComputer对象，相同的一个Fermionic计算机
    """
    def copy(self):
        return copy.deepcopy(self)