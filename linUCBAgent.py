import numpy as np

class LinUCBAgent():
    # Performance: 0.4311969839773798

    def __init__(self, alpha, action_dim, feature_dim):
        self.A = []
        self.b = []
        for a in range(action_dim):
            self.A.append(np.eye(feature_dim))
            self.b.append(np.zeros(feature_dim))
        self.alpha = alpha
        self.action_dim = action_dim
        self.feature_dim = feature_dim
        self.confidence_intervals = []
    
    def predict(self, x):
        intervals = np.zeros((self.action_dim, 1))
        theta = np.zeros((self.action_dim, self.feature_dim))
        for a in range(self.action_dim):
            A_inv = np.linalg.inv(self.A[a])
            theta[a, :] = np.dot(A_inv, self.b[a])
            intervals[a] = self.alpha * np.sqrt(np.dot(np.dot(x.T, A_inv), x))
        self.confidence_intervals.append(intervals)
        scores = np.dot(theta, x) + intervals
        return np.argmax(scores)

    def update_reward(self, r, a, x):
        self.A[a] = self.A[a] + np.dot(x, x.T)
        self.b[a] = self.b[a] + r * x