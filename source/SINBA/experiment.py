# Experiment
import optuna
from .residual_block import residual_block

import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from wandb.integration.keras import WandbMetricsLogger
import gc

def objective(trial, X_train, X_val, y_train, y_val, activation_f = ["relu", "relu6", "leaky_relu"],
              layers_interval = (15,20), units_interval = (128,256), regularizer_interval = (1e-7, 1e-5),
              opt = ["sgd", "adam", "rmsprop", "adamw"], eta_interval = (2.5e-4, 1e-3,), class_weight = {},
              dropout_interval = (0.1, 0.15), custom_optimizer = False):
    """
    Function to elaborate the Optuna experiments to help select parameter for the ResNet model.
    """

    tf.keras.backend.clear_session()

    inputs = layers.Input(shape = (X_train.shape[1],))
    
    #############################################################################################################
    
    # Optuna suggests activation function for all layers
    activation = trial.suggest_categorical("Activation_Function", activation_f)
    
    # Optuna suggests regularizer L2 value
    regularizer = trial.suggest_categorical("Layer_Regularizer", ["l1", "l2"])
    reg_value = trial.suggest_float("Regularizer_Value", regularizer_interval[0], regularizer_interval[1], log = True)
    
    # Optuna suggest the number of layers
    n_layers = trial.suggest_int("N_layers", layers_interval[0], layers_interval[1])
    
    # Optuna suggests learning rate value and an optimizer
    lr = trial.suggest_float("learning_rate", eta_interval[0], eta_interval[1],log = True)

    if custom_optimizer == False:
        optimizer_name = trial.suggest_categorical("optimizer", opt)
                                    
        if optimizer_name == "sgd":
            optimizer = tf.keras.optimizers.SGD(learning_rate = lr)
        elif optimizer_name == "adam":
            optimizer = tf.keras.optimizers.Adam(learning_rate = lr)
        elif optimizer_name == "rmsprop":
            optimizer = tf.keras.optimizers.RMSprop(learning_rate = lr)
        else:
            optimizer = tf.keras.optimizers.AdamW(learning_rate = lr)

    else:
        optimizer = custom_optimizer(learning_rate = lr)
    
    #############################################################################################################
    
    # Optuna suggest number of neurons for the first layer

    N = trial.suggest_int("N_1st_layer", units_interval[0], units_interval[1])
    
    x = layers.Dense(N, input_shape = (X_train.shape[1],))(inputs)
    x = layers.Activation(activation)(x)
    x = layers.BatchNormalization()(x)
    
    # Optuna suggests neurons for the residual blocks and if using Dropout block
    
    dropout_per_layer = []    
    dropping_out = trial.suggest_categorical("Dropout", ["y", "n"])

    regularizer = trial.suggest_categorical("Regularizer", ["l1", "l2"])
    reg_value = trial.suggest_float("Reg_value", regularizer_interval[0], regularizer_interval[1])

    Units_per_layer = []
    
    for i in range(n_layers):
        
        n = trial.suggest_int(f"N_{i+1}_layer", units_interval[0], units_interval[1])
        Units_per_layer.append(n)
        
        # i-th residual block:
        
        # Choosing between Dropout or a regulizer
        
        if dropping_out == "y":
            dropout_rate = trial.suggest_float(f"Dropout_value_L{i+2}", dropout_interval[0], dropout_interval[1])
            dropout_per_layer.append(dropout_rate)
            x = residual_block(x = x, perceptrons = n, activation = activation, dropout_rate = dropout_rate,
                               reg_value = reg_value, regularizer = regularizer)
        else:
            dropout_per_layer.append(0.0)
            x = residual_block(x = x, perceptrons = n, activation = activation, dropout_rate = 0.0,
                               reg_value = reg_value, regularizer = regularizer)            
            
    x = layers.Dropout(0.4)(x)  
    outputs = layers.Dense(1, activation = "sigmoid")(x)
    model = models.Model(inputs, outputs)
                              
    model.compile(optimizer = optimizer,
                  loss = "binary_crossentropy",
                  metrics = ["accuracy",
                             tf.keras.metrics.Precision(),
                             tf.keras.metrics.AUC(curve = "ROC"),
                             tf.keras.metrics.AUC(curve = "PR")])
    
    #############################################################################################################
    
    """
    Callbacks
    """
    early_stopping = EarlyStopping(monitor = "val_precision", patience = 10, restore_best_weights = True)
    lr_reduction = ReduceLROnPlateau(monitor = "val_loss", factor = 0.1, patience = 5)
    
    #############################################################################################################
    
    try:
        if class_weight != {}:

            print(model.summary())
        
            history = model.fit(
                X_train, y_train,
                validation_data = (X_val, y_val),
                batch_size = 32,
                epochs = 200,
                verbose = 1, 
                callbacks = [early_stopping, lr_reduction],
                class_weight = class_weight
            )

            val_precision = max(history.history["val_precision"])
        else: 
            print(model.summary())
        
            history = model.fit(
                X_train, y_train,
                validation_data = (X_val, y_val),
                batch_size = 32,
                epochs = 200,
                verbose = 1, 
                callbacks = [early_stopping, lr_reduction],
            )

            val_precision = max(history.history["val_precision"])
        
    except tf.errors.ResourceExhaustedError as e:
        
        print(f"Intento {trial.number} falló debido a: {e}")
        
        tf.keras.backend.clear_session()
        gc.collect()
        
        return float("inf")

    except Exception as e:
        
        print(f"Intento {trial.number} falló. Unexpected error: {e}")
        
        tf.keras.backend.clear_session()
        gc.collect()
        
        return float("inf")
        
    score = val_precision
        
    tf.keras.backend.clear_session()
    gc.collect()

    return 1-score

def objective_tracked(trial, X_train, X_val, y_train, y_val, project_name,
                      activation_f = ["relu", "relu6", "leaky_relu"],layers_interval = (15,20),
                      units_interval = (128,256), regularizer_interval = (1e-7, 1e-5),
                      opt = ["sgd", "adam", "rmsprop", "adamw"], eta_interval = (2.5e-4, 1e-3,),
                      class_weight = {}, dropout_interval = (0.1, 0.15), custom_optimizer = False):
    """
    Function to elaborate the Optuna experiments to help select parameter for the ResNet model.
    It implements WandB tracking.
    """

    tf.keras.backend.clear_session()

    inputs = layers.Input(shape = (X_train.shape[1],))
    
    #############################################################################################################
    
    # Optuna suggests activation function for all layers
    activation = trial.suggest_categorical("Activation_Function", activation_f)
    
    # Optuna suggests regularizer L2 value
    regularizer = trial.suggest_categorical("Layer_Regularizer", ["l1", "l2"])
    reg_value = trial.suggest_float("Regularizer_Value", regularizer_interval[0], regularizer_interval[1], log = True)
    
    # Optuna suggest the number of layers
    n_layers = trial.suggest_int("N_layers", layers_interval[0], layers_interval[1])
    
    # Optuna suggests learning rate value and an optimizer
    lr = trial.suggest_float("learning_rate", eta_interval[0], eta_interval[1],log = True)

    if custom_optimizer == False:
        optimizer_name = trial.suggest_categorical("optimizer", opt)
                                    
        if optimizer_name == "sgd":
            optimizer = tf.keras.optimizers.SGD(learning_rate = lr)
        elif optimizer_name == "adam":
            optimizer = tf.keras.optimizers.Adam(learning_rate = lr)
        elif optimizer_name == "rmsprop":
            optimizer = tf.keras.optimizers.RMSprop(learning_rate = lr)
        else:
            optimizer = tf.keras.optimizers.AdamW(learning_rate = lr)

    else:
        optimizer = custom_optimizer(learning_rate = lr)
    
    #############################################################################################################
    
    # Optuna suggest number of neurons for the first layer

    N = trial.suggest_int("N_1st_layer", units_interval[0], units_interval[1])
    
    x = layers.Dense(N, input_shape = (X_train.shape[1],))(inputs)
    x = layers.Activation(activation)(x)
    x = layers.BatchNormalization()(x)
    
    # Optuna suggests neurons for the residual blocks and if using Dropout block
    
    dropout_per_layer = []    
    dropping_out = trial.suggest_categorical("Dropout", ["y", "n"])

    regularizer = trial.suggest_categorical("Regularizer", ["l1", "l2"])
    reg_value = trial.suggest_float("Reg_value", regularizer_interval[0], regularizer_interval[1])

    Units_per_layer = []
    
    for i in range(n_layers):
        
        n = trial.suggest_int(f"N_{i+1}_layer", units_interval[0], units_interval[1])
        Units_per_layer.append(n)
        
        # i-th residual block:
        
        # Choosing between Dropout or a regulizer
        
        if dropping_out == "y":
            dropout_rate = trial.suggest_float(f"Dropout_value_L{i+2}", dropout_interval[0], dropout_interval[1])
            dropout_per_layer.append(dropout_rate)
            x = residual_block(x = x, perceptrons = n, activation = activation, dropout_rate = dropout_rate, reg_value = reg_value)
        else:
            dropout_per_layer.append(0.0)
            x = residual_block(x = x, perceptrons = n, activation = activation, dropout_rate = 0.0, reg_value = reg_value)            
            
    x = layers.Dropout(0.4)(x)  
    outputs = layers.Dense(1, activation = "sigmoid")(x)
    model = models.Model(inputs, outputs)
                              
    model.compile(optimizer = optimizer,
                  loss = "binary_crossentropy",
                  metrics = ["accuracy",
                             tf.keras.metrics.Precision(),
                             tf.keras.metrics.AUC(curve = "ROC"),
                             tf.keras.metrics.AUC(curve = "PR")])
    
    wandb.init(
        project = project_name,
        name = f"Trial_{trial.number}",
        reinit = True,
        config = {
            "Units_1": N,
            "Units_per_layer": Units_per_layer,
            "activation": activation,
            "n_layers": n_layers,
            "regularizer": regularizer,
            "reg_value": reg_value,
            "Dropout": dropping_out, 
            "dropout_percentage_per_layer": dropout_rate,
            "learning_rate": lr,
            "optimizer": optimizer_name}
            )
    
    #############################################################################################################
    
    """
    Callbacks
    """
    early_stopping = EarlyStopping(monitor = "val_precision", patience = 10, restore_best_weights = True)
    lr_reduction = ReduceLROnPlateau(monitor = "val_loss", factor = 0.1, patience = 5)
    
    #############################################################################################################
    
    try:
        if class_weight != {}:

            print(model.summary())
        
            history = model.fit(
                X_train, y_train,
                validation_data = (X_val, y_val),
                batch_size = 32,
                epochs = 200,
                verbose = 1, 
                callbacks = [WandbMetricsLogger(log_freq = 5), early_stopping, lr_reduction],
                class_weight = class_weight
            )

            val_precision = max(history.history["val_precision"])
        else: 
            print(model.summary())
        
            history = model.fit(
                X_train, y_train,
                validation_data = (X_val, y_val),
                batch_size = 32,
                epochs = 200,
                verbose = 1, 
                callbacks = [WandbMetricsLogger(log_freq = 5), early_stopping, lr_reduction],
            )

            val_precision = max(history.history["val_precision"])
        
    except tf.errors.ResourceExhaustedError as e:
        
        print(f"Intento {trial.number} falló debido a: {e}")
        
        tf.keras.backend.clear_session()
        wandb.finish()
        gc.collect()
        
        return float("inf")

    except Exception as e:
        
        print(f"Intento {trial.number} falló. Unexpected error: {e}")
        
        tf.keras.backend.clear_session()
        wandb.finish()
        gc.collect()
        
        return float("inf")
        
    score = val_precision
        
    tf.keras.backend.clear_session()
    wandb.finish()
    gc.collect()

    return 1-score
