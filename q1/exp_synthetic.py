import os
import sys
import itertools
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras import optimizers, losses

project_root = os.path.abspath(os.path.join(os.getcwd(), "."))
sys.path.append(project_root)
from model.featureless import MainNetwork  




'''helper fuctions''' 
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


def load_or_calc_save_freq_data_tf(file_path, X, Y):
    """
    If freq file exists, load it; otherwise compute by calc_freq_tf and save as CSV.
    Returns: tf.Tensor
    """
    if os.path.exists(file_path):
        data_np = pd.read_csv(file_path).values
        data = tf.convert_to_tensor(data_np, dtype=tf.float32)
    else:
        data = calc_freq_tf(X, Y)
        pd.DataFrame(data.numpy()).to_csv(file_path, index=False)
    return data


def log_likelihood_tf(out, y, safe_log=0.0):
    """
    out: probabilities, shape (N, D)
    y  : one-hot or multi-hot indicators, shape (N, D)
    """
    ones_indices = tf.equal(y, 1.0)
    probabilities = tf.boolean_mask(out, ones_indices)  # (M,)
    negative_log_prob = -tf.math.log(probabilities + safe_log)
    total_neg_log = tf.reduce_sum(negative_log_prob)
    return total_neg_log / tf.cast(tf.shape(y)[0], out.dtype)


def param_count(depth, width):
    return (depth - 1) * (width ** 2 + width) + 42 * width


def count_params(depth, width):
    return (depth - 1) * (width ** 2 + width) + 42 * width




'''Main training function'''
def train_deephalo_synthetic_tf(budget_name,
                                loss_name,
                                num_epochs,
                                depth,
                                reswidth,
                                batch_size,
                                input_dim,
                                lr,
                                patience,
                                num_params):
    """
    loss_name: 'MSE' or 'NLL'
    num_epochs: int
    depth: int
    reswidth: int (resnet_width)
    batch_size: int
    input_dim: int (opt_size)
    lr: learning rate
    patience: int or None
    num_params: parameter count (for logging)
    """
    
    #build model
    block_types = ['qua'] * (depth - 1)
    main_network = MainNetwork(opt_size=input_dim,
                               depth=depth,
                               resnet_width=reswidth,
                               block_types=block_types)

    #load data
    train_file_path = os.path.join(project_root, 'data', 'Synthetic_20-15-80_Train.csv')
    test_file_path = os.path.join(project_root, 'data', 'Synthetic_20-15-20_Test.csv')

    df_train = pd.read_csv(train_file_path)
    df_test = pd.read_csv(test_file_path)

    X_columns = [col for col in df_train.columns if col.startswith('X')]
    Y_columns = [col for col in df_train.columns if col.startswith('Y')]

    X_train_np = df_train[X_columns].values.astype(np.float32)
    Y_train_np = df_train[Y_columns].values.astype(np.float32)
    X_test_np = df_test[X_columns].values.astype(np.float32)
    Y_test_np = df_test[Y_columns].values.astype(np.float32)

    X_train = tf.convert_to_tensor(X_train_np)
    Y_train = tf.convert_to_tensor(Y_train_np)
    X_test = tf.convert_to_tensor(X_test_np)
    Y_test = tf.convert_to_tensor(Y_test_np)

    X_train_eval = X_train
    Y_train_eval = Y_train

    X_test_eval = X_test
    Y_test_eval = Y_test

    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
    train_dataset = train_dataset.shuffle(buffer_size=X_train.shape[0]).batch(batch_size)

    #optimizer
    optimizer = optimizers.Adam(learning_rate=lr)

    #loss function
    if loss_name == 'NLL':
        def loss_fn(out, target):
            return log_likelihood_tf(out, target)
    else:
        mse = losses.MeanSquaredError()
        def loss_fn(out, target):
            return mse(out, target)

    in_sample_loss = None
    out_sample_loss = None

    in_sample_loss_list = []
    out_sample_loss_list = []
    step_loss_list = []

    best_loss = float('inf')
    epochs_without_improvement = 0

    logs = (f"Training model with Depth {depth} and Width {reswidth} for up to {num_epochs} epochs.\n")
    print(logs)

    # Build model once (necessary before saving weights)
    dummy = tf.zeros((1, input_dim), dtype=tf.float32)
    main_network(dummy, training=False)

    for epoch in range(num_epochs):
        for data_batch, target_batch in tqdm(train_dataset,
                                             desc=f"Epoch {epoch + 1}/{num_epochs}",
                                             leave=False):
            with tf.GradientTape() as tape:
                output_batch, _ = main_network(data_batch, training=True)
                loss_batch = loss_fn(output_batch, target_batch)

            grads = tape.gradient(loss_batch, main_network.trainable_variables)
            optimizer.apply_gradients(zip(grads, main_network.trainable_variables))

            step_rmse = float(tf.sqrt(loss_batch).numpy())
            step_loss_list.append(step_rmse)

        #evaluation on full train/test 
        in_sample_output, _ = main_network(X_train_eval, training=False)
        out_sample_output, _ = main_network(X_test_eval, training=False)

        in_sample_loss = float(tf.sqrt(loss_fn(in_sample_output,
                                               Y_train_eval)).numpy())
        out_sample_loss = float(tf.sqrt(loss_fn(out_sample_output,
                                                Y_test_eval)).numpy())

        in_sample_loss_list.append(in_sample_loss)
        out_sample_loss_list.append(out_sample_loss)

        if (epoch + 1) % 50 == 0:
            logs = (f'Epoch [{epoch + 1}/{num_epochs}], Original RMSE: {in_sample_loss:.4f}')
            print(logs)

        #early stopping 
        if patience is not None:
            if in_sample_loss < best_loss - 1e-5:
                best_loss = in_sample_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break
    
    os.makedirs('./result', exist_ok=True)
    with open(f'./result/depth{depth}_{budget_name[:-1]}.pkl', 'wb') as f:
        pickle.dump(in_sample_loss_list, f)

    #save weights
    save_dir = os.path.join(project_root, 'model', 'Synthetic')
    os.makedirs(save_dir, exist_ok=True)
    weights_path = os.path.join(save_dir,
                                f"Depth-{depth}-Width-{reswidth}.weights.h5")
    main_network.save_weights(weights_path)
    print(f"Saved weights to: {weights_path}")

    #summary
    model_hyperpara = {
        'depth': depth,
        'resnet_width': reswidth,
        'epoch': num_epochs,
        'batch_size': batch_size,
        'lr': lr,
        'loss_name': loss_name,
        'para_num': num_params,
    }

    result = {
        'insample_rmse': in_sample_loss,
        'outsample_rmse': out_sample_loss,
    }

    print("Model hyperparams:", model_hyperpara)
    print("Result:", result)

    #return
    loss_record = {
        'insample_rmse_list': in_sample_loss_list,
        'outsample_rmse_list': out_sample_loss_list,
        'insample_step_rmse_list': step_loss_list,
    }

    return model_hyperpara, result, loss_record




if __name__ == "__main__":

    Loss_Name = 'MSE'
    epochs = 500
    dim = 20
    batch_size = 1024
    learning_rate = 0.0001
    patience = None   #or an integer, e.g. 20

    budget = 500000
    tolerance = 0.003  # ±0.3%
    min_params = int(budget * (1 - tolerance))
    max_params = int(budget * (1 + tolerance))

    candidates = []
    for depth in range(3, 8):  # must have at least input/output layers
        for width in range(100, 1000):
            p = count_params(depth, width)
            if min_params <= p <= max_params:
                candidates.append((depth, width, p))

    candidates.sort(key=lambda x: x[0])  # sort by depth

    for d, w, p in candidates[:10]:
        print('depth', d, 'width', w, 'params', p)

    experiments = {
        '200k': [(3, 306), (4, 251), (5, 218), (6, 195), (7, 179)],
        '500k': [(7, 285), (6, 312), (5, 348), (4, 401), (3, 489)],
    }

    for budget_name, configs in experiments.items():
        print(f'=== Testing Budget Group: {budget_name} ===')
        for depth, width in configs:
            num_params = param_count(depth, width)
            print(f'Depth: {depth}, Width: {width} -> Parameter Num: {num_params}')
            model_hyperpara, result, loss_record = train_deephalo_synthetic_tf(
                budget_name,
                Loss_Name,
                epochs,
                depth,
                width,
                batch_size,
                dim,
                learning_rate,
                patience,
                num_params=num_params,
            )

    #plot
    depths = [3, 4, 5, 6, 7]
    plt.figure(figsize=(6, 4))

    blue_cmap = plt.cm.Blues
    for i, depth in enumerate(depths):
        color = blue_cmap(0.3 + 0.7 * i / len(depths))  
        with open(f'./result/depth{depth}_200.pkl', 'rb') as f:
            loss = pickle.load(f)
        plt.plot(loss, color=color, label=f'200k Dep {depth}', linewidth=1.3)

    orange_cmap = plt.cm.Oranges
    for i, depth in enumerate(depths):
        color = orange_cmap(0.3 + 0.7 * i / len(depths)) 
        with open(f'./result/depth{depth}_500k.pkl', 'rb') as f:
            loss = pickle.load(f)
        plt.plot(loss, color=color, label=f'500k Dep {depth}', linewidth=1.3)

    plt.xlabel("Epochs")
    plt.ylabel("Training RMSE")
    plt.legend()
    plt.tight_layout()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(f'./result/Halo-Effect_model_depth_on_error_depth_all.jpg')
