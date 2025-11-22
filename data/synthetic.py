import numpy as np
import itertools
import pandas as pd

np.random.seed(20)

def generate_probability_list(binary_subset):
    indices_of_ones = [i for i, value in enumerate(binary_subset) if value == 1]
    if not indices_of_ones:
        return [0.0] * len(binary_subset)
    num_ones = len(indices_of_ones)
    probabilities_for_ones = np.random.dirichlet(np.ones(num_ones))
    probability_list = [0.0] * len(binary_subset)
    for i, index in enumerate(indices_of_ones):
        probability_list[index] = probabilities_for_ones[i]
    return probability_list

def generate_one_hot_batch(probabilities, num_samples):
    probabilities = np.array(probabilities)
    p_index = np.random.choice(len(probabilities), size=num_samples, p=probabilities)
    one_hot_batch = np.zeros((num_samples, len(probabilities)))
    one_hot_batch[np.arange(num_samples), p_index] = 1
    return one_hot_batch

def generate_data(p_offerset, p_max_size, p_min_size, num_samples_per_subset_train, num_samples_per_subset_test):
    X = []
    Y = []
    probability_lists = []
    binary_subsets = []

    for r in range(p_min_size, p_max_size + 1):
        for subset in itertools.combinations(p_offerset, r):
            binary_subset = [1 if x in subset else 0 for x in p_offerset]
            probability_list = generate_probability_list(binary_subset)
            probability_lists.append(probability_list)
            binary_subsets.append(binary_subset)

    Y_train = [generate_one_hot_batch(p, num_samples_per_subset_train) for p in probability_lists]
    Y_test = [generate_one_hot_batch(p, num_samples_per_subset_test) for p in probability_lists]

    Y_train = np.concatenate(Y_train, axis=0)
    Y_test = np.concatenate(Y_test, axis=0)

    X_train = [subset for subset in binary_subsets for _ in range(num_samples_per_subset_train)]
    X_test = [subset for subset in binary_subsets for _ in range(num_samples_per_subset_test)]

    X_train = np.array(X_train)
    X_test = np.array(X_test)
    return X_train, Y_train, X_test, Y_test


g_num_product = 20
train_size = 80
test_size = 20
g_offer_set = [i for i in range(g_num_product)]

g_max_assortment_size = 15
g_min_assortment_size = 15


g_X_train, g_Y_train, g_X_test, g_Y_test = generate_data(g_offer_set, g_max_assortment_size, g_min_assortment_size, train_size, test_size)


dataset_train = np.hstack((g_X_train, g_Y_train))
dataset_test = np.hstack((g_X_test, g_Y_test))

columns = ['X' + str(i) for i in g_offer_set] + ['Y' + str(i) for i in g_offer_set]


df_train = pd.DataFrame(dataset_train, columns=columns)
df_test = pd.DataFrame(dataset_test, columns=columns)

df_train.to_csv('./data/Synthetic_20-15-80_Train.csv', index=False)
df_test.to_csv('./data/Synthetic_20-15-20_Test.csv', index=False)
