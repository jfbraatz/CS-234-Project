import numpy as np

class LinUCBHybridAgent():
    # Performance: 0.63289

    def __init__(self, alpha, num_actions, d, k):
        self.A_0 = np.identity(k)
        self.b_0 = np.zeros((k, 1))
        self.A = []
        self.B = []
        self.b = []
        for a in range(num_actions):
            self.A.append(np.identity(d))
            self.B.append(np.zeros((d, k)))
            self.b.append(np.zeros((d, 1)))

        self.alpha = alpha
        self.num_actions = num_actions
        self.d = d
        self.k = k
    
    def predict(self, x, z=None):
        if z == None:
            z = x
            
        beta = np.dot(np.linalg.inv(self.A_0), self.b_0)
        upper_confidence_bounds = []
        for a in range(self.num_actions):
            A_0_inv = np.linalg.inv(self.A_0)
            A_a_inv = np.linalg.inv(self.A[a])

            theta = np.dot(A_a_inv, (
                self.b[a] - np.dot(self.B[a], beta)))

            s_terms = [
                    [z.T, A_0_inv, z],
                    [-2*z.T, A_0_inv, self.B[a].T, A_a_inv, x],
                    [x.T, A_a_inv, x],
                    [x.T, A_a_inv, self.B[a], A_0_inv, self.B[a].T, 
                        A_a_inv, x]
                    ]
            s = sum([np.linalg.multi_dot(elements)
                for elements in s_terms])

            p = np.dot(z.T, beta) + np.dot(x.T, theta) + self.alpha * np.sqrt(s)
            upper_confidence_bounds.append(p)

        return np.argmax(upper_confidence_bounds)

    def update_reward(self, r, a, x, z=None):
        if z == None:
            z = x

        BT_A_inv = np.dot(self.B[a].T, np.linalg.inv(self.A[a]))
        A_0_delta = np.dot(BT_A_inv, self.B[a])
        b_0_delta = np.dot(BT_A_inv, self.b[a])

        self.A_0 += A_0_delta
        self.b_0 += b_0_delta
        self.A[a] += np.dot(x, x.T)
        self.B[a] += np.dot(x, z.T)
        self.b[a] += r * x
        self.A_0 += np.dot(z, z.T) - A_0_delta
        self.b_0 += r * z - b_0_delta
