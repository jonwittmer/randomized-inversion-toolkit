import numpy as np
def isScalar(val):
    if (val * np.array([1])).size == 1:
        return True
    else:
        return False
