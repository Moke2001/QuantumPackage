import galois
import numpy as np
from Physics.QuantumComputation.Code.ClassicalCode.LinearCode import LinearCode


class EuclideanCode(LinearCode):
    #%%  USER：构造函数
    """""
    self.number_checker：int对象，校验子的数目
    self.number_bit：int对象，bit的数目
    self.check_matrix：np.array of GF(2)对象，校验矩阵
    self.rank：int对象，校验矩阵的秩
    self.distance：int对象，线性码的码距
    self.codeword_matrix：np.array of GF(2)对象，生成矩阵
    self.dimension：int对象，EG维度
    self.prime：int对象，EG素数
    self.power：int对象，EG幂
    """""
    def __init__(self,dimension,power,prime=2):
        self.dimension=dimension
        self.prime=prime
        self.power=power
        H_dual = self.generate_matrix()
        assert np.all(np.mod(H_dual@H_dual.T,2)==0)
        super().__init__(H_dual)


    #%%  KEY：生成Euclid LDPC code
    """
    output：np.array of GF(2)对象，奇偶校验矩阵
    """
    def generate_matrix(self):

        ##  计算相关的参数
        q=self.prime**self.power  # 坐标值的个数
        number_point = q ** self.dimension  # 点的个数
        GF = galois.GF(q ** self.dimension,repr='power')  # 用q**m有限域表示m维GF(q)上的几何点
        a = GF.primitive_element  # 几何有限域的基元
        J = int((q ** (self.dimension - 1) - 1) // (q - 1))  # 线的数目

        ##  计算所有的循环族的直线集合
        number_class=0
        line_matrix=np.empty((J,number_point-1,q),dtype=type(a))  # 直线集合构成循环族的集合
        for i in range(number_point-1):
            point_b=a**i  # 直线斜率
            line_vector=np.empty((number_point-1,q),dtype=type(point_b))  # 直线集合构成的循环族

            ##  生成第一条直线
            for j in range(q):
                point_temp=a+GF.elements[j]*point_b
                line_vector[0,j]=point_temp
            line_vector[0,:]=np.sort(line_vector[0,:])

            ##  要求不过原点
            if line_vector[0,0]==GF.elements[0]:
                continue

            ##  求直线族中其他直线
            for j in range(1,number_point-1):
                line_vector[j,:]=[(a**j)*temp for temp in line_vector[0,:]]
                line_vector[j,:]=np.sort(line_vector[j,:])

            ##  判断生成的直线族是否重复
            for j in range(number_class):
                flag=False
                for k in range(number_point-1):
                    if line_vector[0,0]== line_matrix[j,k,0]:
                        if line_vector[0,1]==line_matrix[j,k,1]:
                            flag=True
                            break
                if flag:
                    break
                if j==number_class-1:
                    line_matrix[number_class,:,:]=line_vector
                    number_class = number_class + 1
            if number_class==0:
                line_matrix[number_class, :, :] = line_vector
                number_class=1

            if number_class==J:
                break

        ##  拼接成校验矩阵
        H_list=[]
        for i in range(number_class):
            H=np.zeros((number_point-1,number_point-1),dtype=int)
            for j in range(number_point-1):
                for k in range(q):
                    element_temp=str(line_matrix[i, j, k])
                    if element_temp=='1':
                        element_temp=0
                    elif element_temp=='0':
                        raise ValueError
                    elif element_temp=='α':
                        element_temp=1
                    else:
                        element_temp=int(element_temp[2::])
                    H[j,int(element_temp)]=1
            H_list.append(H)

        ##  构造完整的校验矩阵
        H_left = np.hstack([H_j.T for H_j in H_list])  # Transposed blocks
        H_right = np.hstack(H_list)  # Original blocks
        H = np.hstack([H_left, H_right])

        return H