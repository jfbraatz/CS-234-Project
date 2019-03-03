import numpy as np
from env import *

class WarfarinBandit():

    def simulate(self):
        X, Y = get_data_npy() # X, Y = get_data_csv('data/train_data.npz')
        num_samples, N = X.shape
        shuffled = np.random.permutation(np.column_stack((Y, X)))
        Y_shuffled = shuffled[:, 0]
        X_shuffled = shuffled[:, 1:]

        print("Simulating", num_samples, "patients...")

        num_correct = 0
        for i in range(num_samples):
            prediction = self.predict(X_shuffled[i, :])
            reward = 1 if prediction == Y_shuffled[i] else 0
            self.update_reward(reward, prediction, X_shuffled[i, :])
            num_correct += reward

        performance = float(num_correct) / num_samples
        print('Performance:', performance)
        return performance

    def predict(self, x):
        pass

    def update_reward(self, r, a, x):
        pass