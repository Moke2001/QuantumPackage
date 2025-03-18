from Physics.QuantumComputation.Code.QuantumCode.Transformer.Css2Majorana import css2majorana
from Physics.QuantumComputation.Code.QuantumCode.Transformer.Majorana2Css import majorana2css
from Physics.QuantumComputation.Code.QuantumCode.Transformer.Pauli2Majorana import pauli2majorana


def map_chain(pauli_code):
    print('当前代码是：Pauli code')
    print('它的纠错距离是：' + str(pauli_code.get_weight()))
    print('它的物理比特数是：' + str(pauli_code.number_fermion))
    print('它的逻辑比特数是：' + str(pauli_code.number_fermion - len(pauli_code.generator_vector)))
    print('')

    majorana_code=pauli2majorana(pauli_code)
    print('当前代码是：Majorana code')
    print('它的纠错距离是：' + str(majorana_code.get_weight()))
    print('它的物理比特数是：' + str(majorana_code.number_fermion))
    print('它的逻辑比特数是：' + str(majorana_code.number_fermion - len(majorana_code.generator_vector)))
    print('')

    css_code=majorana2css(majorana_code)
    print('当前代码是：CSS code')
    print('它的纠错距离是：' + str(css_code.get_weight()))
    print('它的物理比特数是：' + str(css_code.number_fermion))
    print('它的逻辑比特数是：' + str(css_code.number_fermion - len(css_code.generator_vector)))
    print('')

    fermionic_code=css2majorana(css_code)
    print('当前代码是：Fermionic code')
    print('它的纠错距离是：' + str(fermionic_code.get_weight()))
    print('它的物理比特数是：' + str(fermionic_code.number_fermion))
    print('它的逻辑比特数是：' + str(fermionic_code.number_fermion - len(fermionic_code.generator_vector)))
    print('')
