from Physics.QuantumComputation.Code.QuantumCode.FermionicCode import FermionicCode


#%%  USER：将两个code联合起来
def code_link(code_0, code_1):

    ##  参数标准化
    assert isinstance(code_0, FermionicCode)
    assert isinstance(code_1, FermionicCode)

    ##  获取关键数据
    code_product=FermionicCode()  # 生成合并结果
    n_0=code_0.number_qubit
    n_1=code_1.number_qubit
    code_product.define_qubit(n_0+n_1)  # 定义code中qubits数目

    ##  加入code_0的稳定子X
    for i in range(len(code_0.check_list)):
        code_product.push(code_0.check_list[i])

    ##  加入code_1的稳定子
    for i in range(len(code_1.check_list)):
        temp=code_1.check_list[i]
        temp.x_vector=temp.x_vector+n_0
        temp.z_vector=temp.z_vector+n_0
        code_product.push(temp)

    for i in range(len(code_0.logical_operator_list_x)):
        code_product.logical_operator_list_x.append(code_0.logical_operator_list_x[i])
    for i in range(len(code_0.logical_operator_list_z)):
        code_product.logical_operator_list_z.append(code_0.logical_operator_list_z[i])
    for i in range(len(code_1.logical_operator_list_x)):
        temp=code_1.logical_operator_list_x[i].copy()
        temp.x_vector=temp.x_vector+n_0
        temp.z_vector=temp.z_vector+n_0
        code_product.logical_operator_list_x.append(temp)
    for i in range(len(code_1.logical_operator_list_z)):
        temp=code_1.logical_operator_list_z[i].copy()
        temp.x_vector=temp.x_vector+n_0
        temp.z_vector=temp.z_vector+n_0
        code_product.logical_operator_list_z.append(temp)

    ##  返回结果
    return code_product