import numpy as np
from MokeGroup.PauliGroup import PauliGroup


class CliffordPauliComputer:
    #%%  USER：构造函数
    """""
    self.number_qubit：int对象，qubit数目
    self.probability：float对象，错误发生几率
    self.psi：np.array of MajoranaGroup对象，态矢的稳定子生成元数组
    """""
    def __init__(self):
        self.number_qubit=None
        self.probability=None
        self.psi=None


    #%%  USER：定义函数
    """""
    input.number_qubit：int对象，qubit数目
    input.probability：float对象，错误发生几率
    influence：修改self.number_qubit和self.probability
    """""
    def define(self,number_qubit,probability):
        assert isinstance(number_qubit,int)
        assert isinstance(probability,float)
        self.number_qubit=number_qubit
        self.probability=probability


    #%%  USER：重载相等运算符
    """""
    input.other：CliffordPauliComputer对象，判断两个态矢是否相等
    output：bool对象，判断结果
    """""
    def __eq__(self, other):
        return PauliGroup.equal(self.psi,other.psi)


    #%%  USER：初始化方法
    """""
    input.params：int对象，初始化参数
    influence：修改self.psi
    """""
    def initialize(self,params):
        if params==1 or params==-1:
            stabilizer_vector=np.empty(self.number_qubit,dtype=PauliGroup)
            for i in range(self.number_qubit):
                stabilizer_temp=PauliGroup()
                data_vector_temp=np.zeros(self.number_qubit*2,dtype=int)
                data_vector_temp[i*2+1]=1
                stabilizer_temp.define(params,data_vector_temp)
                stabilizer_vector[i]=stabilizer_temp
            self.psi=stabilizer_vector
        else:
            raise NotImplementedError('不支持的初始化参数')


    #%%  USER：执行Pauli-X量子逻辑门
    """""
    input.index：int对象，量子门作用的qubit的序号
    influence：将第index个qubit上的Pauli-X门作用在self.psi上面，修改self.psi
    """""
    def X(self,index):
        for i in range(self.psi.shape[0]):
            s_z=self.psi[i][index*2+1]
            if s_z==1:
               self.psi[i].coff=self.psi[i].coff*(-1)


    #%%  USER：执行Pauli-Y量子逻辑门
    """""
    input.index：int对象，量子门作用的qubit的序号
    influence：将第index个qubit上的Pauli-Y门作用在self.psi上面，修改self.psi
    """""
    def Y(self,index):
        for i in range(self.psi.shape[0]):
            s_x = self.psi[i][index * 2]
            s_z = self.psi[i][index * 2 + 1]
            if not (s_z == 1 and s_x == 0):
                self.psi[i].coff = self.psi[i].coff * (-1)


    #%%  USER：执行Pauli-Z量子逻辑门
    """""
    input.index：int对象，量子门作用的qubit的序号
    influence：将第index个qubit上的Pauli-Z门作用在self.psi上面，修改self.psi
    """""
    def Z(self,index):
        for i in range(self.psi.shape[0]):
            s_x = self.psi[i][index * 2]
            if s_x == 1:
                self.psi[i].coff = self.psi[index].coff * (-1)


    #%%  USER：执行Hadamard量子逻辑门
    """""
    input.index：int对象，量子门作用的qubit的序号
    influence：将第index个qubit上的Hadamard门作用在self.psi上面，修改self.psi
    """""
    def H(self,index):
        for i in range(self.psi.shape[0]):
            s_x = self.psi[i][index * 2]
            s_z = self.psi[i][index * 2 + 1]
            if s_x==1 and s_z==0:
                self.psi[i][index*2+1]=1
                self.psi[i][index * 2]=0
            elif s_x==0 and s_z==1:
                self.psi[i][index*2+1]=0
                self.psi[i][index * 2]=1
            elif s_x==1 and s_z==1:
                self.psi[i].coff = self.psi[index].coff * (-1)


    #%%  USER：执行CX量子逻辑门
    """""
    input.index：int对象，量子门作用的qubit的序号
    influence：以index_control作为控制位，以index_target，用CX门修改self.psi
    """""
    def CX(self, index_control, index_target):
        for i in range(self.psi.shape[0]):
            s_x_control = self.psi[i][index_control*2]
            s_z_control = self.psi[i][index_control*2 + 1]
            s_x_target = self.psi[i][index_target*2]
            s_z_target = self.psi[i][index_target*2 + 1]
            temp=[0,0,0,0]
            if s_x_control==1:
                temp[0]=1
                temp[2]=1
            if s_z_control==1:
                temp[1]=1
            if s_x_target==1:
                temp[2]=np.mod(1+temp[2],2)
            if s_z_target==1:
                temp[1]=np.mod(1+temp[1],2)
                temp[3]=1
            self.psi[i][index_control*2]=temp[0]
            self.psi[i][index_control*2 + 1]=temp[1]
            self.psi[i][index_target*2]=temp[2]
            self.psi[i][index_target*2 + 1]=temp[3]


    #%%  USER：执行S量子逻辑门
    """""
    input.index：int对象，量子门作用的qubit的序号
    influence：将第index个qubit上的S门作用在self.psi上面，修改self.psi
    """""
    def S(self,index):
        for i in range(self.psi.shape[0]):
            s_x = self.psi[i][index * 2]
            s_z = self.psi[i][index * 2 + 1]
            if s_x==1:
                self.psi[i].coff=self.psi[index].coff*1j
                self.psi[i][index * 2+1]=np.mod(s_z+1,2)


    #%%  USER：执行测量
    """""
    input.index：int对象，测量的量子比特
    output：int对象，测量结果
    influence：在第index个qubit上测量，使self.psi坍缩
    """""
    def measure(self,index):
        flag=-1
        for i in range(self.psi.shape[0]):
            s_x = self.psi[i][index * 2]
            if s_x==1 and flag==-1:
                flag=i
            elif s_x==1 and flag!=-1:
                self.psi[i]=self.psi[i]*self.psi[flag]
        if flag==-1:
            return 1
        else:
            if np.random.rand()<0.5:
                coff_new=1
                pauli_vector_new=np.zeros_like(self.psi[0].pauli_vecctor,dtype=int)
                pauli_vector_new[index*2+1]=1
                stabilizer_new=PauliGroup()
                stabilizer_new.define(pauli_vector_new,coff_new)
                self.psi[flag]=stabilizer_new
                return 1
            else:
                coff_new=-1
                pauli_vector_new=np.zeros_like(self.psi[0].pauli_vecctor,dtype=int)
                pauli_vector_new[index*2+1]=1
                stabilizer_new = PauliGroup()
                stabilizer_new.define(pauli_vector_new,coff_new)
                self.psi[flag]=stabilizer_new
                return 1


    #%%  USER：发生量子错误
    """""
    input.args：int or list of int对象，发生量子错误的qubits的序号
    influence：按self.probability几率在相应的qubits上发生量子错误，修改self.psi
    """""
    def noise(self,*args):
        for i in range(len(args)):
            sample_temp=np.random.rand()
            if sample_temp<self.probability:
                self.X(args[i])
            elif self.probability<sample_temp<self.probability*2:
                self.Y(args[i])
            elif self.probability*2<sample_temp<self.probability*3:
                self.Z(args[i])
