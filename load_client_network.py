import tensorflow as tf
import numpy as np
import pickle as pkl

from custom_objects import Alsing, CustomTanh, create_msre_loss

def load_model_and_scalers(cosmo_model):
    model_path = f'/Users/andreas/Documents/Codes/notebooks/likelihood sampling with CLiENT networks/results_it9_20260112/{cosmo_model}/trained_models/trained_model_it_9.keras'
    x_scaler_path = f'/Users/andreas/Documents/Codes/notebooks/likelihood sampling with CLiENT networks/results_it9_20260112/{cosmo_model}/scalers/x_scaler_it_9.pkl'
    y_scaler_path = f'/Users/andreas/Documents/Codes/notebooks/likelihood sampling with CLiENT networks/results_it9_20260112/{cosmo_model}/scalers/y_scaler_it_9.pkl'

    with open(x_scaler_path, 'rb') as f:
        x_scaler = pkl.load(f)
    with open(y_scaler_path, 'rb') as f:
        y_scaler = pkl.load(f)

    x_mean = tf.constant(x_scaler.mean_, dtype=tf.float32)
    x_scale = tf.constant(x_scaler.scale_, dtype=tf.float32)
    y_mean = tf.constant(y_scaler.mean_, dtype=tf.float32)
    y_scale = tf.constant(y_scaler.scale_, dtype=tf.float32)

    mean_square_relative_error = create_msre_loss(y_global_max=10.0, kappa=3.0, n=27.0, y_std=y_scale)
    custom_objects={'Alsing': Alsing, 'CustomTanh': CustomTanh, 'mean_square_relative_error': mean_square_relative_error}

    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    
    ## define box function that is exponentially increasing outside limits. It should be auto-differentiable.
    lower = tf.constant([
        0, 0, 0, 0, 0, 0.004, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.9
    ], dtype=tf.float32)

    upper = tf.constant([
        3, 0.5, 2, 5, 2, 1, 0.8, 3.0, 200, 1, 10, 400, 400, 400, 400, 10, 50, 50, 100, 400, 10, 10, 10, 10, 10, 10, 3000, 3000, 1.1
    ], dtype=tf.float32)

    if 'lcdm' in cosmo_model:
        # remove indices 6 and 7 from lower and upper bounds for LCDM model
        lower = tf.concat([lower[:6], lower[8:]], axis=0)
        upper = tf.concat([upper[:6], upper[8:]], axis=0)

    @tf.function(reduce_retracing=True)
    def log_prob_fn(x):
        inp = (x - x_mean) / x_scale
        y_scaled = model(inp)
        log_like = tf.reshape(y_scaled * y_scale + y_mean, [-1])
        return log_like
    
    return log_prob_fn, lower, upper
