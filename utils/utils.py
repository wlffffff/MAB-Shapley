import numpy as np
import copy
import random

def exp_details(args):
    print('\nExperimental details:')
    print(f'    Method     : {args.method}')
    print(f'    Dataset     : {args.dataset}')
    print(f'    Model     : {args.model}')
    print(f'    Num of users     : {args.num_users}')
    print(f'    Learning rate  : {args.lr}')
    print(f'    Global Rounds   : {args.epochs}\n')

    print('    Federated parameters:')
    if args.iid:
        print('    IID')
    else:
        print('    Non-IID')
    if args.method == "mab_n_stratified_shapley" or args.method == "mab_mc_shapley" :
        print(f'    Para_H        : {args.para_H}')
        print(f'    Para_K        : {args.para_K}')
    if args.method == "mab_mc_shapley" :
        print(f'    Iteration        : {args.iteration}')
    # print(f'    Fraction of users  : {args.frac}')
    print(f'    Local Batch size   : {args.bs_train}')
    print(f'    Local Epochs       : {args.local_ep}\n')
    return

def worst_fraction(acc_list, fraction):
    acc_sort = sorted(acc_list, reverse=False)
    worst = [acc_sort[i] for i in range(int(len(acc_list)*fraction))]
    return sum(worst)/len(worst)

def best_fraction(acc_list, fraction):
    acc_sort = sorted(acc_list, reverse=True)
    best = [acc_sort[i] for i in range(int(len(acc_list)*fraction))]
    return sum(best)/len(best)

def add_noise(args, y_train, dict_users):
    np.random.seed(args.seed)

    gamma_s = [1]*int(args.level_n_system * args.num_users) +[0]*int((1-args.level_n_system) * args.num_users)# np.random.binomial(1, args.level_n_system, args.num_users)
    np.random.shuffle(gamma_s)
    gamma_c_initial = np.random.rand(args.num_users)
    gamma_c_initial = (1 - args.level_n_lowerb) * gamma_c_initial + args.level_n_lowerb
    gamma_c = gamma_s * gamma_c_initial

    y_train_noisy = copy.deepcopy(y_train)

    real_noise_level = np.zeros(args.num_users)
    for i in np.where(gamma_c > 0)[0]:
        sample_idx = np.array(list(dict_users[str(i)]))
        prob = np.random.rand(len(sample_idx))
        noisy_idx = np.where(prob <= gamma_c[i])[0]
        y_train_noisy[sample_idx[noisy_idx]] = np.random.randint(0, 10, len(noisy_idx))
        noise_ratio = np.mean(y_train[sample_idx] != y_train_noisy[sample_idx])
        print("Client %d, noise level: %.4f (%.4f), real noise ratio: %.4f" % (i, gamma_c[i], gamma_c[i] * 0.9, noise_ratio))
        real_noise_level[i] = noise_ratio
    return (y_train_noisy, gamma_s, real_noise_level)