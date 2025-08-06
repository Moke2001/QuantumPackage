import numpy as np


def rate_no(number):
    physical_error_rate=np.linspace(0.001,0.2,40)
    logical_error_rate=np.zeros(40)
    for i in range(len(physical_error_rate)):
        p=physical_error_rate[i]
        logical_error_rate[i]=1-(1-p)**(number)
    return logical_error_rate

if __name__ == '__main__':
    print(rate_no(4))
