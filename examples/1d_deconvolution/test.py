import numpy as np
class base:
    param = None
    def __init__(self):
        self.param = 2
        self.other = np.array([1])
        
class child(base):
    def __init__(self):
        super().__init__()
        print(self.param)
        print(self.other)
        
a = child()
