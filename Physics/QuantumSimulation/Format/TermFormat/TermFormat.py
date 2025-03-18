class TermFormat:
    #%%  USER：构造函数
    """""
    Term对象的构造函数
    self.label：str对象，作用量标签
    self.effect：str对象，作用量类型
    self.time：bool对象，判断作用量是否含时
    self.strength：float对象，当作用量不含时时代表强度
    self.function：function对象，当作用量含时时代表随时间变化方式
    self.function_params：dict对象，当作用量含时时代表含参函数的参数
    """""
    def __init__(self, label,effect,*args):
        self.label = label
        self.effect = effect
        if len(args)==1:
            self.time=False
            self.strength=args[0]
        elif len(args,)==2:
            self.time=True
            self.function=args[0]
            self.function_params=args[1]
        elif len(args,)==0:
            self.empty=True
        else:
            raise ValueError("Wrong number of arguments")
        
        
    #%%  KEY：判断形式满足性
    def fit(self, label,*args):
        return self.label == label


    #%%  KEY：获得作用量标签
    def get_label(self):
        return self.label


    #%% KEY：获得作用量强度
    def get_strength(self):
        if self.time:
            return self.function,self.function_params
        else:
            return self.strength


    #%%  KEY：更改作用量强度
    def change(self,*args):
        if len(args)==1:
            self.time=False
            self.strength=args[0]
        elif len(args)==2:
            self.time=True
            self.function=args[0]
            self.function_params=args[1]
        else:
            raise ValueError("Wrong number of arguments")
