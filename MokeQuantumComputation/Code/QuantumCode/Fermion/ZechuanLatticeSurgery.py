import numpy as np
from MokeQuantumComputation.Code.QuantumCode.Fermion.FermionicCode import FermionicCode
from MokeQuantumComputation.Code.QuantumCode.Fermion.FermionicColorCode import FermionicColorCode
from MokeQuantumComputation.Code.QuantumCode.Fermion.MajoranaOperator import MajoranaOperator


#%%  USER：生成Fermionic lattice surgery之后的code
"""
input.code_0：FermionicCode对象
input.code_1：FermionicCode对象
input.index_0：int对象，逻辑算符下标
input.index_1：int对象，逻辑算符下标
ouput：FermionicCode对象，融合结果
"""
def ZechuanLatticeSurgery(code_ldpc, index)->FermionicCode:


    #%%  SECTION：数据标准化
    code_ldpc=code_ldpc.copy()
    assert isinstance(code_ldpc, FermionicCode)
    assert isinstance(index, int)


    #%%  SECTION：数据预处理
    number_qubit_ldpc=code_ldpc.number_qubit
    logical_operator_ldpc=code_ldpc.logical_operator_list_x[index]
    support_index_vector_ldpc = np.array(logical_operator_ldpc.x_vector, dtype=int)
    code_color=FermionicColorCode(len(support_index_vector_ldpc))
    number_qubit_color=code_color.number_qubit
    logical_operator_color = code_color.logical_operator_list_x[0]
    code_color.index_map(number_qubit_ldpc+number_qubit_color,np.arange(number_qubit_color)+number_qubit_ldpc)
    code_ldpc.index_map(number_qubit_ldpc+number_qubit_color,np.arange(number_qubit_ldpc))
    support_index_vector_color=np.array(logical_operator_color.x_vector,dtype=int)
    support_index_vector_color=support_index_vector_color.tolist()
    support_index_vector_ldpc=support_index_vector_ldpc.tolist()

    ##  提取与这些费米子相关联的校验子
    check_list_x_fix_ldpc=[]
    check_list_x_unfix_ldpc=[]
    check_list_z_ldpc=[]
    check_list_x_fixed_ldpc=[]
    check_list_x_fix_color=[]
    check_list_x_unfix_color=[]
    check_list_z_color=[]
    check_list_x_fixed_color=[]

    for i in range(len(code_ldpc.check_list)):
        if len(code_ldpc.check_list[i].x_vector)>0:
            if len(set(code_ldpc.check_list[i].x_vector)&set(support_index_vector_ldpc))>0:
                check_list_x_fix_ldpc.append(code_ldpc.check_list[i].copy())
            else:
                check_list_x_unfix_ldpc.append(code_ldpc.check_list[i].copy())
        else:
            check_list_z_ldpc.append(code_ldpc.check_list[i].copy())

    for i in range(len(code_color.check_list)):
        if len(code_color.check_list[i].x_vector)>0:
            if len(set(code_color.check_list[i].x_vector)&set(support_index_vector_color))>0:
                check_list_x_fix_color.append(code_color.check_list[i].copy())
            else:
                check_list_x_unfix_color.append(code_color.check_list[i].copy())
        else:
            check_list_z_color.append(code_color.check_list[i].copy())

    code=FermionicCode()
    code.define_qubit(number_qubit_ldpc+number_qubit_color)

    ##  记录与support关联的ancilla的索引
    ancilla_dict_ldpc={}
    ancilla_dict_color={}
    for i in range(len(support_index_vector_ldpc)):
        ancilla_dict_ldpc[support_index_vector_ldpc[i]]=[]
        ancilla_dict_color[support_index_vector_color[i]]=[]

    ##  对color code索引
    check_index_color=code_color.logical_plaqutte=[1,0,2,3]
    single_plaqutte=[0,1,2,3]
    def find_plaqutte(index_0,index_1):
        op_0=check_list_x_fix_color[0]
        op_1=check_list_x_fix_color[1]
        op_2=check_list_x_fix_color[2]
        op_3=check_list_x_fix_color[3]
        if index_0==0 and index_1==1:
            return op_1,[1],None
        elif index_0==0 and index_1==2:
            return op_1.mul(op_0,code.number_qubit),[0,1],None
        elif index_0==0 and index_1==3:
            return op_1.mul(op_2,code.number_qubit),[1,2],[support_index_vector_color[1],support_index_vector_color[2]]
        elif index_0==0 and index_1==4:
            return op_1.mul(op_3,code.number_qubit),[1,3],[support_index_vector_color[1],support_index_vector_color[3]]
        elif index_0==1 and index_1==2:
            return op_0,[0],None
        elif index_0==1 and index_1==3:
            return op_0.mul(op_2,code.number_qubit),[0,2],None
        elif index_0==1 and index_1==4:
            return op_0.mul(op_3,code.number_qubit),[0,3],[support_index_vector_color[2],support_index_vector_color[3]]
        elif index_0==2 and index_1==3:
            return op_2,[2],None
        elif index_0==2 and index_1==4:
            return op_2.mul(op_3,code.number_qubit),[2,3],None
        elif index_0==3 and index_1==4:
            return op_3,[3],None

    check_list_gauge=[]
    ##  对support组队
    couple_list=[]
    for i in range(len(check_list_x_fix_ldpc)):
        temp=set(check_list_x_fix_ldpc[i].x_vector) & set(support_index_vector_ldpc)
        temp=list(temp)
        couple_list.append([])
        for j in range(len(temp)//2):
            couple_list[i].append((temp[2*j],temp[2*j+1]))

    ##  修改LDPC code的check
    for i in range(len(check_list_x_fix_ldpc)):
        for j in range(len(couple_list[i])):
            x_vector_temp=check_list_x_fix_ldpc[i].x_vector.copy()
            z_vector_temp=check_list_x_fix_ldpc[i].z_vector.copy()
            code.push_qubit(1)
            ancilla_dict_ldpc[couple_list[i][j][0]].append((code.qubit_list[-1],'x'))
            ancilla_dict_ldpc[couple_list[i][j][1]].append((code.qubit_list[-1],'z'))
            x_vector_temp=np.append(x_vector_temp,code.qubit_list[-1])
            z_vector_temp=np.append(z_vector_temp,code.qubit_list[-1])
            check_list_x_fix_ldpc[i]=MajoranaOperator(x_vector_temp, z_vector_temp,1)
            index_0=couple_list[i][j][0]
            index_1=couple_list[i][j][1]
            index_min=support_index_vector_ldpc.index(index_0)
            index_max=support_index_vector_ldpc.index(index_1)
            if index_min>index_max:
                temp=index_min
                index_min=index_max
                index_max=temp
            operator_color_temp,index_list,associate_list=find_plaqutte(index_min,index_max)
            x_vector_temp = operator_color_temp.x_vector.copy()
            z_vector_temp = operator_color_temp.z_vector.copy()
            for i in range(len(index_list)):
                single_plaqutte[index_list[i]]=None
            if associate_list is not None:
                index_temp_0=associate_list[0]
                index_temp_1=associate_list[1]
                code.push_qubit(2)

                ancilla_dict_color[index_temp_0].append((code.qubit_list[-1],'x'))
                ancilla_dict_color[index_temp_0].append((code.qubit_list[-2],'x'))
                ancilla_dict_color[index_temp_1].append((code.qubit_list[-1],'z'))
                ancilla_dict_color[index_temp_1].append((code.qubit_list[-2],'z'))
                x_vector_temp_gauge=[code.qubit_list[-2],code.qubit_list[-1]]
                z_vector_temp_gauge=[code.qubit_list[-2],code.qubit_list[-1]]
                check_list_gauge.append(MajoranaOperator(x_vector_temp_gauge, z_vector_temp_gauge, 1))
                x_vector_temp=np.append(x_vector_temp,code.qubit_list[-2])
                z_vector_temp=np.append(z_vector_temp,code.qubit_list[-2])
            index_0_color = support_index_vector_color[support_index_vector_ldpc.index(index_0)]
            index_1_color = support_index_vector_color[support_index_vector_ldpc.index(index_1)]
            code.push_qubit(1)
            x_vector_temp=np.append(x_vector_temp,code.qubit_list[-1])
            z_vector_temp=np.append(z_vector_temp,code.qubit_list[-1])
            ancilla_dict_color[index_0_color].append((code.qubit_list[-1],'x'))
            ancilla_dict_color[index_1_color].append((code.qubit_list[-1],'z'))
            check_list_x_fixed_color.append(MajoranaOperator(x_vector_temp, z_vector_temp,1))
            if associate_list is not None:
                x_vector_temp=[code.qubit_list[-4],code.qubit_list[-1]]
                z_vector_temp=[code.qubit_list[-4],code.qubit_list[-1]]
            else:
                x_vector_temp=[code.qubit_list[-2],code.qubit_list[-1]]
                z_vector_temp=[code.qubit_list[-2],code.qubit_list[-1]]
            check_list_gauge.append(MajoranaOperator(x_vector_temp, z_vector_temp,1))

    for i in range(len(single_plaqutte)):
        if single_plaqutte[i] is not None:
            operator_color_temp=check_list_x_fix_color[single_plaqutte[i]]
            overlap=list(set(operator_color_temp.x_vector)&set(support_index_vector_color))
            x_vector_temp = operator_color_temp.x_vector.copy()
            z_vector_temp = operator_color_temp.z_vector.copy()
            code.push_qubit(2)
            ancilla_dict_color[overlap[0]].append((code.qubit_list[-1], 'x'))
            ancilla_dict_color[overlap[0]].append((code.qubit_list[-2], 'x'))
            ancilla_dict_color[overlap[1]].append((code.qubit_list[-1], 'z'))
            ancilla_dict_color[overlap[1]].append((code.qubit_list[-2], 'z'))
            x_vector_temp=np.append(x_vector_temp,code.qubit_list[-2])
            z_vector_temp=np.append(z_vector_temp,code.qubit_list[-2])
            check_list_x_fixed_color.append(MajoranaOperator(x_vector_temp, z_vector_temp, 1))
            x_vector_temp = [code.qubit_list[-2], code.qubit_list[-1]]
            z_vector_temp = [code.qubit_list[-2], code.qubit_list[-1]]
            check_list_gauge.append(MajoranaOperator(x_vector_temp, z_vector_temp, 1))
    check_list_measurement = []
    for i in range(len(support_index_vector_ldpc)):
        x_vector_temp=[support_index_vector_ldpc[i],support_index_vector_color[i]]
        z_vector_temp=[]
        for j in range(len(ancilla_dict_ldpc[support_index_vector_ldpc[i]])):
            if ancilla_dict_ldpc[support_index_vector_ldpc[i]][j][1]=='x':
                x_vector_temp=np.append(x_vector_temp,ancilla_dict_ldpc[support_index_vector_ldpc[i]][j][0])
            else:
                z_vector_temp = np.append(z_vector_temp, ancilla_dict_ldpc[support_index_vector_ldpc[i]][j][0])
        for j in range(len(ancilla_dict_color[support_index_vector_color[i]])):
            if ancilla_dict_color[support_index_vector_color[i]][j][1]=='x':
                x_vector_temp=np.append(x_vector_temp,ancilla_dict_color[support_index_vector_color[i]][j][0])
            else:
                z_vector_temp = np.append(z_vector_temp, ancilla_dict_color[support_index_vector_color[i]][j][0])
        check_list_measurement.append(MajoranaOperator(x_vector_temp, z_vector_temp, 1))
    check_list_x_fixed_ldpc=check_list_x_fix_ldpc
    code.check_list=check_list_x_unfix_ldpc+check_list_x_unfix_color
    code.check_list+=check_list_z_ldpc+check_list_z_color
    code.check_list+=check_list_x_fixed_color+check_list_x_fixed_ldpc
    code.check_list+=check_list_measurement
    code.check_list+=check_list_gauge
    code.number_checker=len(code.check_list)

    #%%  SECTION：返回结果

    assert code.commute_judge()  # 检查对易性
    return code