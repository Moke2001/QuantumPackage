import galois
import numpy as np


#%%  USER：GF(2)上的扩展格拉姆-施密特正交化算法
"""
intput.vectors：list of np.array of GF(2)对象，向量基矢
output：list of np.array of GF(2)对象，正交基向量列表
"""

def orthogonalize(vectors):


    #%%  SECTION：正交化方法

    def gf2_gram_schmidt(vectors, bilinear_form):
        o_i = None
        GF2 = galois.GF2
        k = len(vectors)
        # 复制向量以避免修改原始数据
        B_i = [v.copy() for v in vectors]
        O = []  # 存储正交基

        for i in range(k):
            # 1. 优先选择非迷向向量
            non_isotropic_found = False
            for idx in range(len(B_i)):
                if bilinear_form(B_i[idx], B_i[idx]) != GF2(0):
                    if idx != 0:
                        B_i[0], B_i[idx] = B_i[idx], B_i[0]
                    non_isotropic_found = True
                    break

            b1 = B_i[0]
            if non_isotropic_found:
                o_i = b1
            else:
                # 2. 尝试构造非迷向向量
                found = False
                for j in range(1, len(B_i)):
                    b_j = B_i[j]
                    if bilinear_form(b1, b_j) != GF2(0):
                        v = b1 + b_j
                        if bilinear_form(v, v) != GF2(0):
                            o_i = v
                            found = True
                            break
                if not found:
                    o_i = b1

            O.append(o_i)

            # 3. 更新剩余向量
            next_B = []
            for j in range(1, len(B_i)):
                b = B_i[j]
                coef = bilinear_form(b, o_i)
                b_new = b + coef * o_i
                next_B.append(b_new)

            B_i = next_B

        return O


    #%%  SECTION：计算双线性形式

    def inner_product(u, v):
        return np.dot(u, v)


    #%%  SECTION：处理正交向量组

    ##  双线性形式矩阵
    bilinear_form1 = lambda u, v: inner_product(u, v)
    ortho_basis1 = gf2_gram_schmidt(vectors, bilinear_form1)
    judge_matrix=np.zeros((len(ortho_basis1), len(ortho_basis1)))
    for i in range(len(ortho_basis1)):
        for j in range(len(ortho_basis1)):
            judge_matrix[i,j] = np.dot(ortho_basis1[i], ortho_basis1[j])

    ##  判断是否存在正交基
    flag_0=np.trace(judge_matrix)
    flag_1=np.count_nonzero(judge_matrix)
    assert flag_0>0
    print("输入向量:", [v.tolist() for v in vectors])
    print("正交基:", [v.tolist() for v in ortho_basis1])

    ##  输出结果
    return ortho_basis1