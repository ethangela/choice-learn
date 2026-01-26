import numpy as np
import itertools
import pandas as pd


def generate_one_hot(probabilities):
    probabilities = np.array(probabilities)
    p_index = np.random.choice(len(probabilities), p=probabilities)
    one_hot = np.zeros_like(probabilities)
    one_hot[p_index] = 1
    return one_hot
    
product_set = [0, 1, 2, 3]
offer_set = [0, 1, 2, 3]
X = []
Y = []
np.random.seed(10)

hypothetical_choice_p = [[0.98, 0.02, 0, 0],
                         [0.5, 0, 0.5, 0],
                         [0.5, 0, 0, 0.5],
                         [0, 0.5, 0.5, 0],
                         [0, 0.5, 0, 0.5],
                         [0, 0, 0.9, 0.1],
                         [0.49, 0.01, 0.5, 0],
                         [0.49, 0.01, 0, 0.5],
                         [0.5, 0, 0.45, 0.05],
                         [0, 0.5, 0.45, 0.05],
                         [0.49, 0.01, 0.45, 0.05]]

index = 0
for r in range(2, len(offer_set) + 1):
    for subset in itertools.combinations(offer_set, r):
        binary_subset = [1 if x in subset else 0 for x in offer_set]
        p = hypothetical_choice_p[index]
        for _ in range(200):
            X.append(binary_subset)
            Y.append(generate_one_hot(p).reshape((1, len(product_set))))
        index += 1


X = np.array(X)
Y = np.concatenate(Y, axis=0)
dataset = np.concatenate((X, Y), axis=1)
df = pd.DataFrame(dataset, columns=['X' + str(i) for i in product_set] + ['Y' + str(i) for i in product_set])
csv_file = './data/hypothetical-4p-test.csv'
df.to_csv(csv_file, index=False)


X = []
Y = []
np.random.seed(42)
index = 0
for r in range(2, len(offer_set) + 1):
    for subset in itertools.combinations(offer_set, r):
        binary_subset = [1 if x in subset else 0 for x in offer_set]
        p = hypothetical_choice_p[index]
        for _ in range(2000):
            X.append(binary_subset)
            Y.append(generate_one_hot(p).reshape((1, len(product_set))))
        index += 1


X = np.array(X)
Y = np.concatenate(Y, axis=0)
dataset = np.concatenate((X, Y), axis=1)
df = pd.DataFrame(dataset, columns=['X' + str(i) for i in product_set] + ['Y' + str(i) for i in product_set])
csv_file = './data/hypothetical-4p-train.csv'
df.to_csv(csv_file, index=False)