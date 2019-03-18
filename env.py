import pandas as pd
import numpy as np
import os.path
import matplotlib.pyplot as plt
import argparse

from LASSOBandit import *
from linUCBHybridAgent import *
from linUCBAgent import *

def get_data_npy(loadfile='data/train_data.npz'):
    # load existing numpy data
    data = np.load(loadfile)
    X_train = data['X_train']
    Y_train = data['Y_train']
    data.close()
    return X_train, Y_train

parser = argparse.ArgumentParser()
parser.add_argument('-a', '--agent', help="baseline, linUCB, hybrid, LASSO", default='linUCB')
parser.add_argument('-d', '--delta', default=0.1, type=float)
parser.add_argument('-p', '--plot', action='store_false')
parser.add_argument('-l', '--log', action='store_true')
parser.add_argument('--seed', default=42, type=int)
parser.add_argument('-t', '--num_test', default=0, type=int)
parser.add_argument('-n', '--num_trials', default=1, type=int)
parser.add_argument('--datafile', default='features.npz')
args = parser.parse_args()

if os.path.isfile(args.datafile): 
    X, Y = get_data_npy(args.datafile)
else:
    print(f'datafile {args.datafile} not found')
    quit()

np.random.seed(args.seed)
shuffled = np.random.permutation(np.column_stack((Y, X)))
Y_train = shuffled[:, 0]
X_train = shuffled[:, 1:]
if args.num_test:
    print(f'Holding out {args.num_test} patients for testing')
    X_test = X_train[-args.num_test:, :]
    Y_test = Y_train[-args.num_test:]
    X_train = X_train[:-args.num_test, :]
    Y_train = Y_train[:-args.num_test]
    test_interval = 500
    test_performances = []
    test_intervals = []

num_train, N = X_train.shape
print('Features size:', N)
action_dim = 3

alpha = 1 + np.sqrt(np.log(2/args.delta)/2)
if args.agent == 'baseline':
    agent = Baseline1()
elif args.agent == 'linUCB':
    agent = LinUCBAgent(alpha, action_dim, N)
elif args.agent == 'hybrid':
    agent = LinUCBHybridAgent(alpha, action_dim, N, N)
elif args.agent == 'LASSO':
    agent = LASSOBandit(action_dim, N, 1, 5, 0.05, 0.05)
else:
    print("Invalid agent")
    quit()

print("Simulating", num_train, "patients...")

regret = []
performances = []
for j in range(args.num_trials):
    print(f"Starting iteration {j}")
    shuffled = np.random.permutation(np.column_stack((Y_train, X_train)))
    Y_train = shuffled[:, 0]
    X_train = shuffled[:, 1:]

    regret.append([0.0])
    if args.num_test:
        test_performances.append([])
        test_intervals.append([])
    for i in range(num_train):
        if args.num_test and i % test_interval == 0:
            test_intervals[j].append(i)
            test_regret = 0
            for i_test in range(args.num_test):
                x_test = np.expand_dims(X_test[i_test, :], axis=1)
                prediction = agent.predict(x_test)
                reward = 0 if prediction == Y_test[i_test] else -1
                test_regret -= reward

            test_performance = 1.0 - test_regret / args.num_test
            test_performances[j].append(test_performance)

        x = np.expand_dims(X_train[i, :], axis=1)
        prediction = agent.predict(x)

        reward = 0 if prediction == Y_train[i] else -1
        regret[j].append(regret[j][-1] - reward)

        if args.agent == 'linUCB':
            agent.update_reward(reward, prediction, x)
        elif args.agent == 'hybrid':
            agent.update_reward(reward, prediction, x, x)
        elif args.agent == 'LASSO':
            agent.update_reward(reward)      

    if args.num_test:
        test_intervals[j].append(i)
        test_regret = 0
        for i_test in range(args.num_test):
            x_test = np.expand_dims(X_test[i_test, :], axis=1)
            prediction = agent.predict(x_test)
            reward = 0 if prediction == Y_test[i_test] else -1
            test_regret -= reward

        test_performance = 1.0 - test_regret / args.num_test
        test_performances[j].append(test_performance)
        print(f'Test performance on final iteration {i} on {args.num_test} patients: {test_performance}')

    performance = 1.0 - regret[j][-1] / num_train
    performances.append(performance)

    print('Performance:', performance)
    print('Total Regret:', regret[j][-1])

regret = np.array(regret)[:, 1:]
avg_regret = np.mean(regret, axis=0).reshape(-1)

incorrect_decisions = regret / (np.arange(num_train) + 1)
avg_incorrect_decisions = np.mean(incorrect_decisions, axis=0)

performances = np.array(performances)
avg_performance = np.mean(performances)

print('---------------------------------')
print(f'Average regret: {avg_regret[-1]}')
print(f'Average performance: {avg_performance}')

if args.plot:
    t = np.arange(num_train)
    plt.plot(t, avg_regret, label='avg regret')
    plt.plot(t, np.sqrt(action_dim * N * t), label='asymtotic bound')
    plt.plot(t, t,label='linear')
    plt.legend(loc='best')
    plt.title(f'Average Regret for {args.agent}')
    plt.show()

    err_min = np.abs(avg_incorrect_decisions - np.min(incorrect_decisions, axis=0))
    err_max = np.abs(avg_incorrect_decisions - np.max(incorrect_decisions, axis=0))
    plt.errorbar(t, avg_incorrect_decisions, yerr=[err_min, err_max], ecolor='lavender', label=args.agent)
    plt.legend(loc='best')
    plt.title(f'Average Incorrect decisions for {args.agent}')
    plt.show()

    if args.num_test:
        test_performances = np.array(test_performances)
        avg_test_performances = np.mean(test_performances, axis=0)
        plt.plot(test_intervals[0], avg_test_performances)
        plt.title(f'Test performance over time for {args.agent}')
        plt.show()

if args.log:
    logfile = 'log/log.txt'
    with open(logfile, 'w') as f:
        for i in range(num_train):
            line = f'reward: {rewards[i]},\t prediction: {predictions[i]}\n'
            f.write(line)





