import numpy as np

class s1fBaseline():

    def __init__(self, feature_dim):
        self.w = np.zeros(feature_dim)
        self.w[:6] = [  -.2546,
                        .0118,
                        .0134,
                        1,
                        1.2799,
                        -.5695]
    
    def predict(self, x):
        p = np.dot(self.w, x)**2 / 7.0
        # print(p)
        if p < 3:
            return 0
        elif p <= 7:
            return 1
        else:
            return 2