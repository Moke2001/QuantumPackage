import numpy as np
import galois
from itertools import product
from MokeQuantumComputation.Code.ClassicalCode.LinearCode import LinearCode


class ProjectiveCode(LinearCode):
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
    def __init__(self,s):
        self.s=s
        H_asym = self.generate_matrix()
        H_dual = np.hstack([H_asym[0:(H_asym.shape[0]-1)//2, 0:2 ** s - 1], H_asym[0:(H_asym.shape[0]-1)//2, 2 ** s:4 ** s+1]])
        assert np.all(H_dual@H_dual.T==0)
        super().__init__(H_dual)

    #%%  KEY：生成Projective LDPC code
    """
    output：np.array of GF(2)对象，奇偶校验矩阵
    """
    def generate_matrix(self):
        def construct_projective_plane(s):
            """构造有限射影平面 PG(2, 2^s)"""
            q = 2 ** s
            GF = galois.GF(2 ** s)

            # 生成点集 [x, y, z] (齐次坐标)
            points = []
            for x, y, z in product(GF.elements, repeat=3):
                if x == 0 and y == 0 and z == 0:
                    continue  # 排除原点

                # 标准化表示: 使第一个非零元素为1
                coords = [x, y, z]
                for i in range(3):
                    if coords[i] != 0:
                        scale = coords[i] ** -1
                        point = tuple(c * scale for c in coords)
                        if point not in points:
                            points.append(point)
                        break
            return points

        def get_lines(points):
            """获取所有直线"""
            lines = []
            for i in range(len(points)):
                for j in range(i + 1, len(points)):
                    p1, p2 = points[i], points[j]
                    # 计算通过两点的直线方程
                    a = p1[1] * p2[2] - p1[2] * p2[1]
                    b = p1[2] * p2[0] - p1[0] * p2[2]
                    c = p1[0] * p2[1] - p1[1] * p2[0]

                    # 标准化直线方程
                    for k in range(3):
                        if [a, b, c][k] != 0:
                            scale = [a, b, c][k] ** -1
                            a, b, c = a * scale, b * scale, c * scale
                            break
                    line = (a, b, c)
                    if line not in lines:
                        lines.append(line)
            return lines

        def construct_incidence_matrix(points, lines):
            """构造点线关联矩阵"""
            n_points = len(points)
            n_lines = len(lines)
            M = np.zeros((n_lines, n_points), dtype=int)

            for i, line in enumerate(lines):
                a, b, c = line
                for j, point in enumerate(points):
                    x, y, z = point
                    if a * x + b * y + c * z == 0:  # 点在直线上
                        M[i, j] = 1
            return M

        def construct_hyperoval(points, s):
            """构造超椭圆 (二次曲线 + 核点)"""
            q = 2 ** s
            GF = galois.GF(2 ** s)

            # 构造二次曲线 (x^2 = yz)
            conic_points = []
            for point in points:
                x, y, z = point
                if x ** 2 == y * z:  # 满足二次曲线方程
                    conic_points.append(point)

            # 计算核点 (所有切线的交点)
            if conic_points:
                # 使用Galois域安全的加法
                x_sum = GF(0)
                y_sum = GF(0)
                z_sum = GF(0)
                for p in conic_points:
                    x_sum += p[0]
                    y_sum += p[1]
                    z_sum += p[2]

                # 标准化核点
                nucleus = (x_sum, y_sum, z_sum)
                for i in range(3):
                    if nucleus[i] != 0:
                        scale = nucleus[i] ** -1
                        nucleus = tuple(c * scale for c in nucleus)
                        break
            else:
                # 默认核点 (当s=1时)
                nucleus = (GF(1), GF(0), GF(0))

            hyperoval = conic_points.copy()
            hyperoval.append(nucleus)
            non_hyperoval = [p for p in points if p not in hyperoval]

            return list(hyperoval), non_hyperoval

        def classify_lines(lines, hyperoval, points):
            """分类直线: secant 或 skew"""
            secant_lines = []
            skew_lines = []

            for idx, line in enumerate(lines):
                # 计算直线与超椭圆的交点
                intersection_count = 0
                for point in hyperoval:
                    x, y, z = point
                    a, b, c = line
                    if a * x + b * y + c * z == 0:  # 点在直线上
                        intersection_count += 1

                if intersection_count == 2:  # secant line
                    secant_lines.append(idx)
                elif intersection_count == 0:  # skew line
                    skew_lines.append(idx)

            return secant_lines, skew_lines
        """构造非对称双包含量子 LDPC 码 C_asym"""
        # 步骤1: 构造射影平面
        s=self.s
        points = construct_projective_plane(s)
        lines = get_lines(points)
        M_pi = construct_incidence_matrix(points, lines)
        n_points = len(points)

        # 步骤2: 构造超椭圆并分类点
        hyperoval, non_hyperoval = construct_hyperoval(points, s)
        non_hyperoval_indices = [points.index(p) for p in non_hyperoval]

        # 步骤3: 分类直线
        secant_lines, skew_lines = classify_lines(lines, hyperoval, points)

        # 步骤4: 构造 H_sk (skew lines)
        if skew_lines:
            H_sk = M_pi[skew_lines][:, non_hyperoval_indices]
            u_vector = np.ones((len(skew_lines), 1), dtype=int)
            H_sk = np.hstack([H_sk, u_vector])
        else:
            H_sk = np.zeros((0, len(non_hyperoval_indices) + 1), dtype=int)

        # 步骤5: 构造 H_se (secant lines)
        if secant_lines:
            H_se = M_pi[secant_lines][:, non_hyperoval_indices]
            u_vector = np.ones((len(secant_lines), 1), dtype=int)
            H_se = np.hstack([H_se, u_vector])
        else:
            H_se = np.zeros((0, len(non_hyperoval_indices) + 1), dtype=int)

        # 步骤6: 构造量子校验矩阵 (分块对角)
        n_sk = H_sk.shape[1] if H_sk.size > 0 else len(non_hyperoval_indices) + 1
        n_se = H_se.shape[1] if H_se.size > 0 else len(non_hyperoval_indices) + 1

        # 创建全零矩阵作为基础
        total_rows = H_sk.shape[0] + H_se.shape[0]
        total_cols = n_sk + n_se
        H_asym = np.zeros((total_rows, total_cols), dtype=int)

        # 填充 H_sk 部分
        if H_sk.size > 0:
            H_asym[:H_sk.shape[0], :H_sk.shape[1]] = H_sk

        # 填充 H_se 部分
        if H_se.size > 0:
            start_row = H_sk.shape[0]
            start_col = n_sk
            H_asym[start_row:start_row+H_se.shape[0], start_col:start_col+H_se.shape[1]] = H_se

        # 转换为伽罗瓦域矩阵
        GF2 = galois.GF(2)
        return GF2(H_asym)