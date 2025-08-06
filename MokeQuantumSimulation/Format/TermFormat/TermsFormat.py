from MokeQuantumSimulation.Format.TermFormat.CouplingTermFormat import CouplingTermFormat
from MokeQuantumSimulation.Format.TermFormat.MultiTermFormat import MultiTermFormat
from MokeQuantumSimulation.Format.TermFormat.OnsiteTermFormat import OnsiteTermFormat
from MokeQuantumSimulation.Format.TermFormat.OverallCouplingTermFormat import OverallCouplingTermFormat
from MokeQuantumSimulation.Format.TermFormat.OverallMultiTermFormat import OverallMultiTermFormat
from MokeQuantumSimulation.Format.TermFormat.OverallOnsiteTermFormat import OverallOnsiteTermFormat
from MokeQuantumSimulation.Format.TermFormat.TermFormat import TermFormat


class TermsFormat:
    #%%  USER：构造函数
    """""
    Terms对象的构造函数，一个空构造函数
    self.terms：list of Term对象，储存模型的作用量
    self.number：int对象，储存作用量个数
    TermList对象初始化后通过调用push函数等再进行实例化
    """""
    def __init__(self,*args):
        ##  SECTION：空构造函数-----------------------------------------------------------
        if len(args) == 0:
            self.terms = []
            self.number=0

        ##  SECTION：标准构造函数----------------------------------------------------------
        elif len(args) == 1 and isinstance(args[0], list):
            self.terms = args[0]
            self.number = len(args[0])

        ##  SECTION：复制构造函数---------------------------------------------------------
        elif len(args) == 1 and isinstance(args[0], TermsFormat):
            self.terms = args[0].terms
            self.number = len(self.terms)
    
    
    #%%  USER：Term对象推送到terms
    def push(self, term:TermFormat):
        assert isinstance(term,TermFormat),'term must be of type TermFormat'
        self.terms.append(term)
        self.number+=1
    
    
    #%%  USER：按terms序号删除Term对象
    def pull(self,index:int):
        self.terms.pop(index)
        self.number-=1
    
    
    #%%  USER：按Term对象内容改变terms中的Term对象
    """
    Terms中的Term对象的基本形式：标序位符值
    OnsiteTerm(label,effect,position,op,strength/[function,function_params])
    CouplingTerm(label,effect,position_0,positon_1,op_0,op_1,strength/[function,function_params])
    MultiTerm(label,effect,position_list,op_list,strength/[function,function_params])
    OverallOnsiteTerm(label,effect,cell_index,op,strength/[function,function_params])
    OverallCouplingTerm(label,effect,cell_index_0,cell_index_1,vector,op_0,op_1,strength/[function,function_params])
    OverallMultiTerm(label,effect,cell_index_list,vector_list,op_list,strength/[function,function_params])
    """
    def change(self,*args):
        type=TermsFormat.judge_type(*args)
        term_index=self.find(*args)  # 寻找输入参数对应Term对象所在terms的序号
        self.terms[term_index]=type(*args)  # 根据输入变化做改变


    #%%  USER：按Term对象内容删除terms中的Term对象
    def remove(self,*args):
        term_index=self.find(*args)  # 寻找输入参数对应Term对象所在terms的序号
        self.terms.pop(term_index)


    #%%  USER：清空terms函数
    def clear(self):
        self.terms = []
        self.number=0


    #%%  KEY：在terms中寻找特定Term对象
    """
    find函数只按照标签、序号和位置寻找特定的Term对象，不考虑强度
    """
    def find(self,*args):
        for i in range(len(self.terms)):
            if self.terms[i].fit(*args):
                return i
        return None


    #%%  KEY：获得作用量列表
    def get_terms(self):
        return self.terms


    #%%  KEY：按terms序号返回Term对象
    def get_term(self,index:int):
        return self.terms[index]
    
    
    #%%  KEY：Term对象类型判断函数
    """
    根据输入参数的形式判断输入的参数对应的Term对象类型，只根据序号、位置的形式来判断
    judge_type是一个静态函数
    """
    @staticmethod
    def judge_type(*args):
        ##  OnsiteTerm
        if isinstance(args[2],tuple) and isinstance(args[3],str):
            return OnsiteTermFormat
        ##  CouplingTerm
        elif isinstance(args[2],tuple) and isinstance(args[3],tuple):
            return CouplingTermFormat
        ##  OverallOnsiteTerm
        elif isinstance(args[2],int) and isinstance(args[3],str):
            return OverallOnsiteTermFormat
        ##  OverallCouplingTerm
        elif isinstance(args[2],int) and isinstance(args[3],int):
            return OverallCouplingTermFormat
        elif isinstance(args[2],list) and isinstance(args[3],list):
            ##  OverallMultiTerm
            if isinstance(args[2][0],tuple):
                return MultiTermFormat
            ##  MultiTerm
            else:
                return OverallMultiTermFormat
        else:
            raise TypeError('Type is not supported')