import numpy as np
from sklearn.linear_model import Lasso, LinearRegression

class LASSOBandit():
    # Performance: 0.6306078147612156

    def __init__(self, K, d, q, h, lambda_1, lambda_2_0):
        self.K = K
        self.d = d
        self.q = q
        self.h = h
        self.lambda_1 = lambda_1
        self.lambda_2 = [lambda_2_0]
        self.X = [['dummy'] * d] # matrix. rows are X_t (context)
        self.Y = ['dummy']
        self.t = 0 # gets incremented in predict()
        self.actions = range(1, K+1)

        # These are dummy values, I'm using 1-based indexing
        # because that's what the paper does. If I try to access them
        # accidentally it'll probably throw an error
        self.T_ = ['dummy']
        self.T__ = ['dummy']
        self.S = ['dummy']

        n_max = 13 # TODO find a less lazy solution to this
        for i in self.actions:
            self.T__.append([set()])
            self.S.append([set()])
            self.T_.append({(2**n-1)*K*q + j
                    for n in range(n_max)
                    for j in range(q*(i-1)+1, q*i+1)})
            

    def beta(self, S, lambd):
        alpha = lambd/2.0
        if alpha < 1e-15:
            print("alpha too small, using linear regression")
            clf = LinearRegression(fit_intercept=False)
        else:
            clf = Lasso(alpha=lambd/2, fit_intercept=False,)
        X = [self.X[i] for i in S]
        Y = [self.Y[i] for i in S]
        X = np.array(X)
        X = X.reshape(X.shape[0], -1)
        clf.fit(X, Y)
        return clf.coef_
    
    def predict(self, X_t):
        self.t += 1
        t = self.t
        if t%100 == 0:
            print(t)
        self.X.append(X_t)
        for i in self.actions:
            if self.t in self.T_[i]:
                pi = i
                break
        else:
            scores = [np.dot(X_t.T, self.beta(self.T__[j][t-1], self.lambda_1)) for j in self.actions]
            maximum = np.max(scores)
            scores.insert(0, 'dummy')
            K_hat = {k for k in self.actions
                    if (scores[k] >= maximum - self.h/2.)}
            #print(K_hat)
            #for i in self.actions:
                #print(i, self.S[i][t-1])
            pi = max([(np.dot(X_t.T, self.beta(self.S[k][t-1],
                self.lambda_2[t-1])), k)
                for k in K_hat])[1]

        for i in self.actions:
            self.S[i].append(self.S[i][-1])
            self.T__[i].append(self.T_[i].intersection(set(range(1, t+1))))

        self.S[pi][t] = self.S[pi][t-1].union({t})
        self.lambda_2.append(self.lambda_2[0] * np.sqrt((np.log(t) + np.log(self.d))/t))
#        for i in self.actions:
#            print(i, self.S[i])


        return (pi - 1)

    def update_reward(self, r):
        self.Y.append(r)
