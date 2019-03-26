import pandas as pd
import numpy as np
import os.path
#import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import argparse

from LASSOBandit import *
from linUCBHybridAgent import *
from linUCBAgent import *
from s1fBaseline import *
from baseline import *
from XGBandit import *

def get_data_npy(loadfile='data/train_data.npz'):
    # load existing numpy data
    data = np.load(loadfile)
    X_train = data['X_train']
    Y_train = data['Y_train']
    data.close()
    return X_train, Y_train

parser = argparse.ArgumentParser()
parser.add_argument('-a', '--agent', action='append', help="fixed_dose, s1f, linUCB, hybrid, LASSO", default=['linUCB'])
parser.add_argument('-d', '--delta', default=0.1, type=float)
parser.add_argument('-p', '--plot', action='store_false')
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


logfile = 'log/log.txt'
with open(logfile, 'a+') as f:
    f.write(f'{args}\n')
    f.write('******************\n')

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

num_train, N = X_train.shape
print('Features size:', N)
print("Simulating", num_train, "patients")
action_dim = 3
print(args.agent)

if args.plot:
    colors = {"fixed_dose":"red", "linUCB":"blue", "hybrid":"green", "LASSO":"purple", "s1f":"black", "XGBandit": "brown"}
    colorse = {"fixed_dose":"lightcoral", "linUCB":"skyblue", "hybrid":"springgreen", "LASSO":"violet", "s1f":"grey", "XGBandit": "peru"}
    regret_fig = plt.figure()
    regret_ax = regret_fig.add_subplot(111)

    performance_fig = plt.figure()
    performance_ax = performance_fig.add_subplot(111)

    if args.num_test:
        test_fig = plt.figure()
        test_ax = test_fig.add_subplot(111)

for agent_name in args.agent:
    test_performances = []
    test_intervals = []
    regret = []
    performances = []
    for j in range(args.num_trials):
        alpha = 1 + np.sqrt(np.log(2/args.delta)/2)
        if agent_name == 'fixed_dose':
            agent = Baseline1()
        elif agent_name == 's1f':
            agent = s1fBaseline(N)
        elif agent_name == 'linUCB':
            agent = LinUCBAgent(alpha, action_dim, N)
        elif agent_name == 'hybrid':
            agent = LinUCBHybridAgent(alpha, action_dim, N, 10)
        elif agent_name == 'LASSO':
            agent = LASSOBandit(action_dim, N, 1, 5, 0.05, 0.05)
        elif agent_name == 'XGBandit':
            agent = XGBandit(action_dim, N, 1, 5, 0.05, 0.05)
        else:
            print("Invalid agent")
            continue

        print(f"Starting iteration {j} for {agent_name}")
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
                # print(f'Test performance on iteration {i} on {args.num_test} patients: {test_performance}')

            x = np.expand_dims(X_train[i, :], axis=1)
            prediction = agent.predict(x)

            reward = 0 if prediction == Y_train[i] else -1
            regret[j].append(regret[j][-1] - reward)

            if agent_name == 'linUCB':
                agent.update_reward(reward, prediction, x)
            elif agent_name == 'hybrid':
                agent.update_reward(reward, prediction, x, x)
            elif agent_name == 'LASSO' or agent_name == 'XGBandit':
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
    print('---------------------------------')

    if args.plot:
        t = np.arange(num_train)
        regret_ax.plot(t, avg_regret, label=agent_name, color=colors[agent_name])

        err = np.std(incorrect_decisions, axis=0)
        performance_ax.errorbar(t[2:], avg_incorrect_decisions[2:], yerr=err[2:], color=colors[agent_name], ecolor=colorse[agent_name], label=agent_name)

        if args.num_test:
            test_performances = np.array(test_performances)
            avg_test_performances = np.mean(test_performances, axis=0)
            test_ax.plot(test_intervals[0], avg_test_performances, label=agent_name, color=colors[agent_name])

        logfile = 'log/log.txt'
        with open(logfile, 'a+') as f:
            f.write(f'{agent_name}\n')
            f.write(f'Regret: {avg_regret[-1]}\n')
            f.write(f'Average performance: {avg_performance}\n')
            f.write(f'Incorrect: {avg_incorrect_decisions[-1]}\n')
            if args.num_test:
                f.write(f'Test: {avg_test_performances[-1]}')
            f.write('----------------------------\n')

if args.plot:
    t = np.arange(num_train)
    regret_ax.plot(t, t,label='linear')
    # plt.plot(t, np.sqrt(action_dim * N * t), label='asymtotic bound')
    regret_ax.legend(loc='best')
    regret_ax.set_title(f'Average Regret')
    regret_ax.set_ylabel(f'Regret (# Incorrect Decisions) ')
    regret_ax.set_xlabel(f't (Patients Seen)')
    regret_fig.savefig('regret.png')

    performance_ax.legend(loc='best')
    performance_ax.set_title(f'Average Incorrect Dosages')
    performance_ax.set_ylabel(f'% Incorrect Dosages')
    performance_ax.set_xlabel(f't (Patients Seen)')
    performance_fig.savefig('performance.png')

    if args.num_test:
        test_ax.set_title(f'Test Performance Over Time')
        test_ax.legend(loc='best')
        test_ax.set_ylabel(f'% Correct on test set')
        test_ax.set_xlabel(f't (Patients Seen)')
        test_fig.savefig('test.png')




