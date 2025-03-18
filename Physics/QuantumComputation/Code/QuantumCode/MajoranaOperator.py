import galois
import numpy as np


class MajoranaOperator:
    def __init__(self,x_vector,z_vector,coff):
        assert isinstance(x_vector,np.ndarray) or isinstance(x_vector,list)
        assert isinstance(z_vector,np.ndarray) or isinstance(z_vector,list)
        assert coff==-1 or coff==1
        self.x_vector=np.array(x_vector,dtype=int)
        self.z_vector=np.array(z_vector,dtype=int)
        self.coff=coff
        weight=len(x_vector)+len(z_vector)
        weight=weight*(weight-1)//2
        if weight%2==0:
            self.factor=1
        else:
            self.factor=1j


    def shift(self,number):
        self.x_vector=self.x_vector+number
        self.z_vector=self.z_vector+number


    def define(self,x_vector,z_vector,coff):
        assert isinstance(x_vector,np.ndarray) or isinstance(x_vector,list)
        assert isinstance(z_vector,np.ndarray) or isinstance(z_vector,list)
        assert coff==-1 or coff==1
        self.x_vector=np.array(x_vector,dtype=int)
        self.z_vector=np.array(z_vector,dtype=int)
        self.coff=coff

    def represent(self,number_qubit):
        GF=galois.GF(2**1)
        res=GF(np.zeros(number_qubit*2,dtype=int))
        res[self.x_vector]=1
        res[self.z_vector+number_qubit]=1
        return res

    def dual(self):
        return MajoranaOperator(self.z_vector,self.x_vector,self.coff)

    def get_matrix(self,number_qubit):
        GF=galois.GF(2**1)
        matrix=GF(np.zeros(number_qubit*2, dtype=int))
        matrix[self.x_vector]=1
        matrix[self.z_vector+number_qubit]=1
        return matrix

    def commute(self,other):
        assert isinstance(other,MajoranaOperator)
        overlap_x=len(np.intersect1d(self.x_vector,other.x_vector))
        overlap_z=len(np.intersect1d(self.z_vector,other.z_vector))
        weight=(len(self.x_vector)+len(self.z_vector))*(len(other.x_vector)+len(other.z_vector))
        judge=overlap_x+overlap_z+weight
        return np.mod(judge,2)==0


    def mul(self,other,number_qubit):
        assert isinstance(other,MajoranaOperator)
        ##  计算相乘结果
        data_0=np.zeros(number_qubit*2,dtype=int)
        data_1=np.zeros(number_qubit*2,dtype=int)
        data_0[self.x_vector*2]=1
        data_0[self.z_vector*2+1]=1
        data_1[other.x_vector*2]=1
        data_1[other.z_vector*2+1]=1

        majorana_vector_new = np.zeros_like(data_0, dtype=int)
        new_coff = self.coff * other.coff
        """
        如果对方的算符是1，将对方的算符依次替换到前面，直到到达对应位置上
        替换过程中如果经过一个1，符号要发生一次变化
        如果对方的算符是0，那么不需要替换
        """
        for i in range(len(data_1)):
            if data_1[i] == 0:
                majorana_vector_new[i] = data_0[i]
            else:
                for j in range(len(data_0) - 1, i, -1):
                    if data_0[j] == 1:
                        new_coff = -new_coff
                if data_0[i] == 0:
                    majorana_vector_new[i]=1
                else:
                    majorana_vector_new[i]=0
        x_vector=np.where(majorana_vector_new[0::2]==1)[0]
        z_vector=np.where(majorana_vector_new[1::2]==1)[0]
        if self.factor==other.factor==1j:
            new_coff=-new_coff
        return MajoranaOperator(x_vector,z_vector,new_coff)

    def copy(self):
        return MajoranaOperator(self.x_vector,self.z_vector,self.coff)

    def __str__(self):
        return str('x')+str(self.x_vector)+str('z')+str(self.z_vector)


if __name__=='__main__':
    x=MajoranaOperator([1,4,6],[1,4,6],1)
    y=MajoranaOperator([4,5],[4,5],-1)
    z=x.mul(y,7)
    pass




