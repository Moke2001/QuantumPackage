import galois
from MokeQuantumComputation.Code.ClassicalCode.LinearCode import LinearCode
import numpy as np
import random


class BicycleCode(LinearCode):
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
    def __init__(self,N,k,M,seed):
        random.seed(seed)
        H, diff_set = self.generate_matrix(N, k, M)
        assert np.all(H@H.T==0)
        super().__init__(H)


    # %%  KEY：生成Bicycle LDPC code
    """
    N: int对象，块长度
    k: int对象，行权重
    M: int对象，目标行数 
    """
    def generate_matrix(self,N, k, M):
        def create_circulant_matrix(difference_set, size):
            """创建基于差集的循环矩阵"""
            matrix = np.zeros((size, size), dtype=int)
            for i in range(size):
                for d in difference_set:
                    j = (i + d) % size
                    matrix[i, j] = 1
            return matrix

        def generate_difference_set(n, k):
            """生成满足唯一差值特性的差集"""
            # 初始化差集
            diff_set = [0]
            # 记录已使用的差值
            used_differences = set()

            while len(diff_set) < k:
                candidate = random.randint(1, n - 1)
                valid = True

                # 检查与现有元素的所有差值是否唯一
                for d in diff_set:
                    diff1 = (candidate - d) % n
                    diff2 = (d - candidate) % n

                    if diff1 in used_differences or diff2 in used_differences:
                        valid = False
                        break

                if valid:
                    # 添加新元素并记录差值
                    diff_set.append(candidate)
                    for d in diff_set[:-1]:
                        diff1 = (candidate - d) % n
                        diff2 = (d - candidate) % n
                        used_differences.add(diff1)
                        used_differences.add(diff2)

            return sorted(diff_set)

        # 验证参数
        if N % 2 != 0:
            raise ValueError("N必须是偶数")
        if k % 2 != 0:
            raise ValueError("k必须是偶数")
        if M >= N / 2:
            raise ValueError("M必须小于N/2")

        n = N // 2  # 循环矩阵大小
        k_half = k // 2  # C的行权重

        # 步骤1: 生成满足唯一差值特性的差集
        difference_set = generate_difference_set(n, k_half)

        # 步骤2: 创建循环矩阵C和其转置
        C = create_circulant_matrix(difference_set, n)
        C_T = C.T

        # 步骤3: 构造H0 = [C, C_T]
        H0 = np.hstack((C, C_T))

        # 步骤4: 删除行以实现均匀列权重
        rows_to_keep = list(range(H0.shape[0]))

        # 计算初始列权重
        col_weights = np.sum(H0, axis=0)

        # 计算需要删除的行数
        rows_to_delete = H0.shape[0] - M

        # 贪心算法删除行，使列权重均匀
        for _ in range(rows_to_delete):
            best_row = -1
            best_variance = float('inf')

            # 尝试删除每一行，找到使列权重方差最小的行
            for row in rows_to_keep:
                # 临时删除该行
                temp_weights = col_weights - H0[row, :]
                variance = np.var(temp_weights)

                if variance < best_variance:
                    best_variance = variance
                    best_row = row

            # 删除最佳行
            rows_to_keep.remove(best_row)
            col_weights -= H0[best_row, :]

        # 创建最终矩阵
        H = H0[rows_to_keep, :]

        # 转换为GF(2)矩阵
        GF2 = galois.GF2
        H_gf2 = GF2(H)

        return H_gf2, difference_set