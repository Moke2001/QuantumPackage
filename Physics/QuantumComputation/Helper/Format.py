import galois
import numpy as np


def matrix_format(matrix):
    GF=galois.GF(2**1)
    if isinstance(matrix,list):
        return GF(np.array(matrix,dtype=int))
    elif isinstance(matrix,np.ndarray):
        return GF(matrix)
    else:
        raise TypeError('Matrix must be of type LinearCode or list')

def list_format(original_list):
    GF=galois.GF(2**1)
    list_now=[]
    for i in range(len(original_list)):
        if isinstance(original_list[i], list):
            list_now.append(GF(np.array(original_list[i], dtype=int)))
        elif isinstance(original_list[i], np.ndarray):
            list_now.append(GF(original_list[i]))
        else:
            raise TypeError('Matrix must be of type LinearCode or list')
    return list_now

def coff_format(coff_vector):
    if isinstance(coff_vector,list):
        coff_vector_now=np.array(coff_vector,dtype=complex)
        return coff_vector_now
    elif isinstance(coff_vector,np.ndarray):
        coff_vector_now=np.array(coff_vector,dtype=complex)
        return coff_vector_now
    else:
        raise TypeError('coff_vector must be of type LinearCode or list')
