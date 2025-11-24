import os
import sys
import csv
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import optimizers, losses

project_root = os.path.abspath(os.path.join(os.getcwd(), "."))
print(project_root)
sys.path.append(project_root)
from model.featureless import MainNetwork  



'''Custom NLL loss'''
def log_likelihood_tf(out, y, safe_log=0.0):
    """
    out: probabilities, shape (N, D)
    y  : one-hot or multi-hot indicators, shape (N, D)
    """
    ones_indices = tf.equal(y, 1.0)
    probabilities = tf.boolean_mask(out, ones_indices)  # (M,)
    negative_log_prob = -tf.math.log(probabilities + safe_log)
    total_neg_log = tf.reduce_sum(negative_log_prob)
    # same normalization as original: / y.shape[0]
    return total_neg_log / tf.cast(tf.shape(y)[0], out.dtype)





'''Helper function: frequency-averaged targets'''
def calc_freq_tf(X, Y):
    """
    X: tf.Tensor, shape (N, d_x)
    Y: tf.Tensor, shape (N, d_y)
    Returns: tf.Tensor, shape (N, d_y), where for each group of identical X rows,
             Y is replaced by the average Y for that group.
    """
    X_np = X.numpy()
    Y_np = Y.numpy()

    unique_X, inverse_indices = np.unique(X_np, axis=0, return_inverse=True)
    new_Y = np.zeros_like(Y_np, dtype=Y_np.dtype)

    for k in range(unique_X.shape[0]):
        mask = (inverse_indices == k)
        avg_y = Y_np[mask].mean(axis=0)
        new_Y[mask] = avg_y

    return tf.convert_to_tensor(new_Y, dtype=Y.dtype)





'''Training function''' 
def train_synthetic_tf(loss_name, num_epochs, dep):
    input_dim = 4
    opt_size = input_dim
    depth = dep
    resnet_width = input_dim
    block_types = ['qua'] * (depth - 1)

    # Build model
    model = MainNetwork(opt_size=opt_size,
                        depth=depth,
                        resnet_width=resnet_width,
                        block_types=block_types)

    # Paths
    train_file_path = os.path.join(project_root, 'data', 'hypothetical-4p-train.csv')
    test_file_path = os.path.join(project_root, 'data', 'hypothetical-4p-test.csv')

    # Load CSVs
    df_train = pd.read_csv(train_file_path)
    df_test = pd.read_csv(test_file_path)

    X_columns = [col for col in df_train.columns if col.startswith('X')]
    Y_columns = [col for col in df_train.columns if col.startswith('Y')]

    # Convert to TF tensors
    X_train = tf.constant(df_train[X_columns].values, dtype=tf.float32)
    Y_train = tf.constant(df_train[Y_columns].values, dtype=tf.float32)
    X_test = tf.constant(df_test[X_columns].values, dtype=tf.float32)
    Y_test = tf.constant(df_test[Y_columns].values, dtype=tf.float32)

    # Frequency-averaged targets
    Y_train_freq = calc_freq_tf(X_train, Y_train)
    Y_test_freq = calc_freq_tf(X_test, Y_test)

    # Optimizer
    optimizer = optimizers.Adam(learning_rate=0.01)

    # Loss selection
    if loss_name == 'NLL':
        def loss_fn(out, target):
            return log_likelihood_tf(out, target)
    else:
        mse = losses.MeanSquaredError()
        def loss_fn(out, target):
            return mse(out, target)

    in_loss = None
    freq_loss_val = None

    # Training loop
    for epoch in range(num_epochs):
        with tf.GradientTape() as tape:
            output, _ = model(X_train, training=True) # output: (N, D)
            loss = loss_fn(output, Y_train)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        in_loss = float(loss.numpy())

        # Frequency loss (no gradient)
        out_no_grad, _ = model(X_train, training=False)
        freq_loss = loss_fn(out_no_grad, Y_train_freq)
        freq_loss_val = float(freq_loss.numpy())

        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], '
                  f'Original RMSE: {in_loss ** 0.5:.4f}, '
                  f'Frequency RMSE: {freq_loss_val ** 0.5:.8f}')

    # Final evaluation on test
    final_output, _ = model(X_test, training=False)
    in_sample_RMSE = in_loss ** 0.5
    in_sample_RMSE_Freq = freq_loss_val ** 0.5

    out_loss = float(loss_fn(final_output, Y_test).numpy())
    out_loss_freq = float(loss_fn(final_output, Y_test_freq).numpy())
    out_sample_RMSE = out_loss ** 0.5
    out_sample_RMSE_Freq = out_loss_freq ** 0.5

    # Save weights
    model_path = os.path.join(project_root, 'model', 'Depth2Model-hyp.weights.h5')
    model.save_weights(model_path)

    return (in_sample_RMSE,
            in_sample_RMSE_Freq,
            out_sample_RMSE,
            out_sample_RMSE_Freq)



if __name__ == "__main__":
    
    '''train parameters'''
    Loss_Name = 'MSE'
    epochs = 20000
    depth = 2

    '''training'''
    in_sample_RMSE, in_sample_RMSE_Freq, out_sample_RMSE, out_sample_RMSE_Freq = train_synthetic_tf(Loss_Name, epochs, depth)
    print('in_sample_RMSE, in_sample_RMSE_Freq, out_sample_RMSE, out_sample_RMSE_Freq')
    print(in_sample_RMSE, in_sample_RMSE_Freq, out_sample_RMSE, out_sample_RMSE_Freq)

    '''estimating the context effects'''
    #4 offers
    optsize = 4
    offer_set = list(range(optsize))
    max_effect_order = 3


    #load the pre-trained model
    block_types_eval = ['qua'] * (depth - 1)
    eval_model = MainNetwork(opt_size=optsize,
                             depth=depth,
                             resnet_width=optsize,
                             block_types=block_types_eval)
    dummy = tf.zeros((1, optsize), dtype=tf.float32)
    eval_model(dummy, training=False)
    model_path = os.path.join(project_root, 'model', 'Depth2Model-hyp.weights.h5')
    eval_model.load_weights(model_path)


    #evaluation_input generation 
    def create_evaluation_tensor(indices, order):
        all_subsets = []
        p_subsets = []
        S = []
        for r in range(1, order + 2):
            subsets = itertools.combinations(indices, r)
            for subset in subsets:
                tensor = np.zeros(len(indices), dtype=np.float32)
                tensor[list(subset)] = 1.0
                all_subsets.append(tensor)
                p_subsets.append(subset)
                if r == order + 1:
                    continue
                S.append(subset)
        concatenated = np.stack(all_subsets, axis=0) # (num_subsets, optsize)
        # Return as tf.Tensor for model input, but keep subsets, S as Python lists
        return tf.constant(concatenated, dtype=tf.float32), p_subsets, S

    evaluation_input, p_subsets, S = create_evaluation_tensor(offer_set, max_effect_order)
    choice_prob, eva = eval_model(evaluation_input, training=False)  # eva: (num_subsets, optsize)
    eva_np = eva.numpy()


    #halo & ratio-halo computations
    def cal_halo(evaluation, subsets, S):
        # evaluation: np.array, shape (num_subsets, optsize)
        key_list = [','.join(str(x) for x in s) for s in S]
        empty_value_list = [{} for _ in S]
        halo_dict = dict(zip(key_list, empty_value_list))
        halo_dict[''] = {}

        for idx, subset in enumerate(subsets):
            subset = list(subset)
            if len(subset) == 1:
                j = subset[0]
                # In original code: evaluation[j, j], relying on ordering of subsets;
                # keep it as-is for exact behavior.
                halo_dict[''][str(j)] = float(evaluation[j, j])
            else:
                for k in range(len(subset)):
                    effective_set = subset[:k] + subset[k+1:]
                    key = ','.join(str(x) for x in effective_set)
                    effect = float(evaluation[idx, subset[k]])
                    # subtract lower-order effects
                    for r in range(1, len(effective_set)):
                        minus_subsets = itertools.combinations(effective_set, r)
                        for sub_s in minus_subsets:
                            effect -= halo_dict[','.join(str(x) for x in sub_s)][str(subset[k])]
                    effect -= halo_dict[''][str(subset[k])]
                    halo_dict[key][str(subset[k])] = effect
        return halo_dict

    def normalize_effect(halo_dict):
        for effect_set in list(halo_dict.keys()):
            if halo_dict[effect_set]:
                avg_effect = np.mean(list(halo_dict[effect_set].values()))
                for influenced_item in list(halo_dict[effect_set].keys()):
                    halo_dict[effect_set][influenced_item] -= avg_effect
        return halo_dict

    def insert_number(numbers_str, new_num):
        numbers_list = list(map(int, numbers_str.split(',')))
        numbers_list.append(int(new_num))
        numbers_list.sort()
        return ','.join(map(str, numbers_list))

    def cal_ratio_halo(competing_set, halo, order):
        ratio_halo_dict_key = [key for key in list(halo.keys()) if len(key.split(',')) < order]
        product_pair = list(itertools.combinations(competing_set, 2))
        product_pair_string = [','.join(str(x) for x in pt) for pt in product_pair]
        empty_value_list = [dict(zip(product_pair_string, [None for _ in product_pair_string]))
                            for _ in ratio_halo_dict_key]
        ratio_halo_dict = dict(zip(ratio_halo_dict_key, empty_value_list))
        for es in list(ratio_halo_dict.keys()):
            es_list = es.split(',') if es != '' else []
            if len(es_list) == order:
                continue
            if es == '':
                for pair, pair_name in zip(product_pair, product_pair_string):
                    x, y = str(pair[0]), str(pair[1])
                    ratio_halo_dict[es][pair_name] = \
                        (halo[''][x] + halo[y][x]) - (halo[''][y] + halo[x][y])
            else:
                for pair, pair_name in zip(product_pair, product_pair_string):
                    x, y = str(pair[0]), str(pair[1])
                    if x in es_list or y in es_list:
                        ratio_halo_dict[es][pair_name] = 0.0
                    else:
                        extra_es_x, extra_es_y = insert_number(es, x), insert_number(es, y)
                        ratio_halo_dict[es][pair_name] = \
                            halo[es][x] - halo[es][y] + halo[extra_es_y][x] - halo[extra_es_x][y]
        return ratio_halo_dict, product_pair_string

    def generate_halo_effect_matrix(p_halo_effect_dict, p_optsize, p_offer_set):
        halo_effect_matrix = np.zeros((p_optsize, len(list(p_halo_effect_dict.keys())) - 1))
        keys_list = list(p_halo_effect_dict.keys())
        for row in p_offer_set:
            for col, kk in enumerate(keys_list):
                if kk == '':
                    continue
                if str(row) not in p_halo_effect_dict[kk]:
                    if len(kk) > 1:
                        halo_effect_matrix[row, col] = 0.0
                    else:
                        halo_effect_matrix[row, col] = p_halo_effect_dict[''][str(row)]
                    continue
                halo_effect_matrix[row, col] = p_halo_effect_dict[kk][str(row)]
        return halo_effect_matrix

    def generate_r_halo_matrix(p_r_halo_effect_dict, p_pair_names):
        p_effect_names = list(p_r_halo_effect_dict.keys())
        halo_effect_matrix = np.zeros((len(p_pair_names), len(p_effect_names)))
        for row, pair in enumerate(p_pair_names):
            for col, kk in enumerate(p_effect_names):
                halo_effect_matrix[row, col] = p_r_halo_effect_dict[kk][pair]
        return halo_effect_matrix

    def process_list(input_list):
        result = []
        for string in input_list:
            parts = string.split(',')
            processed_parts = []
            for part in parts:
                try:
                    num = int(part)
                    processed_parts.append(str(num + 1))
                except ValueError:
                    processed_parts.append(part)
            if processed_parts:
                result.append(','.join(processed_parts))
        return result

    halo_effect_dict = cal_halo(eva_np, p_subsets, S)
    halo_effect_dict = normalize_effect(halo_effect_dict)
    halo_matrix = generate_halo_effect_matrix(halo_effect_dict, optsize, offer_set)
    r_halo_dict, pair_names = cal_ratio_halo(offer_set, halo_effect_dict, max_effect_order)
    r_halo_matrix = generate_r_halo_matrix(r_halo_dict, pair_names)

    effect_name_halo = [','.join(str(letter) for letter in name) for name in S]
    effect_name_halo_r = process_list(list(r_halo_dict.keys())[:-1] + ['$∅$'])
    pair_names = process_list(pair_names)


    #output the heatmap
    os.makedirs('./result', exist_ok=True)
    plt.figure(figsize=(12, 8))
    cmap = plt.cm.RdBu
    cmap.set_under('white')
    plt.imshow(r_halo_matrix, cmap=cmap, interpolation='nearest', aspect='auto')
    plt.xticks(range(len(effect_name_halo_r)), effect_name_halo_r, fontsize=16)
    plt.yticks(range(len(pair_names)), pair_names, fontsize=16)
    plt.colorbar()
    plt.xlabel('Effect Source Set (T)', fontsize=18)
    plt.ylabel('Competing Product Pair (j, k)', fontsize=18)
    plt.savefig('./result/Halo-Ratio_Effect.jpg')

