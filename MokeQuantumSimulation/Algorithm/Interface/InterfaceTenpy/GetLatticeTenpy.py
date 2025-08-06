from tenpy import Lattice, LegCharge, Site
from MokeQuantumSimulation.Format.ModelFormat.ModelFormat import ModelFormat


#%%  KEY：返回tenpy晶格对象
"""
input.model：ModelFormat对象，模型
output：Lattice对象，tenpy晶格
influence：本函数不改变参数对象
"""
def get_lattice_tenpy(model_format:ModelFormat)->Lattice:
    ##  SECTION：标准化-----------------------------------------------------------------------------
    assert isinstance(model_format,ModelFormat),'参数model必须是ModelFormat对象'

    ##  SECTION：获取参数---------------------------------------------------------------------------
    cell_period_list=model_format.cell_period_list
    cell_vector_list=model_format.cell_vector_list
    inner_coordinate_list=model_format.site_vector_list

    ##  SECTION：获取元胞tenpy格点------------------------------------------------------------------
    site_tenpy_list=[]
    for i in range(len(model_format.site_dimension_list)):

        ##  根据维度定义Site对象
        site_temp=Site(LegCharge.from_trivial(model_format.site_dimension_list[i]))

        ##  添加算符
        for key in model_format.local_operator_dictionary.keys():
            if not site_temp.valid_opname(key):
                if model_format.site_dimension_list[i]==model_format.local_operator_dictionary[key].shape[0]:
                    site_temp.add_op(key,model_format.local_operator_dictionary[key])
            else:
                if model_format.site_dimension_list[i]==model_format.local_operator_dictionary[key].shape[0]:
                    site_temp.remove_op(key)
                    site_temp.add_op(key,model_format.local_operator_dictionary[key])
        site_tenpy_list.append(site_temp)

    ##  SECTION：返回结果---------------------------------------------------------------------------
    bc=model_format.periodicity
    if isinstance(bc,bool):
        if bc:
            bc_tenpy=['periodic']*len(model_format.cell_period_list)
            return Lattice(cell_period_list, site_tenpy_list, basis=cell_vector_list, positions=inner_coordinate_list,bc=bc_tenpy)
        else:
            return Lattice(cell_period_list, site_tenpy_list, basis=cell_vector_list, positions=inner_coordinate_list)
    elif isinstance(bc,list):
        bc_tenpy=[]
        for i in range(len(bc)):
            if bc[i]:
                bc_tenpy.append('periodic')
            else:
                bc_tenpy.append('open')
        return Lattice(cell_period_list, site_tenpy_list, basis=cell_vector_list, positions=inner_coordinate_list, bc=bc_tenpy)
    else:
        raise TypeError('周期类型有误')