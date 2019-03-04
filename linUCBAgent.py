import numpy as np

class LinUCBAgent():

    def __init__(self, alpha, action_dim, feature_dim):
        self.A = []
        self.b = []
        for a in range(action_dim):
            self.A.append(np.identity(feature_dim))
            self.b.append(np.zeros((feature_dim, 1)))

        self.alpha = alpha
        self.action_dim = action_dim
        self.confidence_intervals = []

    def predict(self, x):
        scores = []
        intervals = []
        for a in range(self.action_dim):
            A_inv = np.linalg.inv(self.A[a])
            theta = np.dot(A_inv,  self.b[a])
            interval = self.alpha * np.sqrt(np.dot(np.dot(x.T, A_inv), x))
            intervals.append(interval)
            p = np.dot(theta.T, x) + interval
            scores.append(p) 

        self.confidence_intervals.append(intervals)
        prediction = np.argmax(scores)
        # print(predictions)
        # if predictions.shape[0] > 1:
        #     prediction = np.random.choice(predictions)
        # else:
        #     prediction = predictions[0]

        return prediction

    def update_reward(self, reward, a, x):
        self.A[a] = self.A[a] + np.dot(x, x.T)
        self.b[a] = self.b[a] + reward * x


