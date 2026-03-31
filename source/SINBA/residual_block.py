# Residual Block
from tensorflow.keras import regularizers, layers

def residual_block(x, perceptrons, activation, dropout_rate = 0.0, reg_value = 1.0e-3, regularizer = "l2"):
    """
    Functions which helps to create the residual block used in the experiments with optuna or the final model.
    It allows the user to configurate the perceptron units per residual layer, the activation function for all the 
    residual block, if it is used Dropout or not, the Dropout rate if so per layer, to choose a layer regularizer,
    and the value of the regularizer.
    """
    residual = x  
    if dropout_rate == 0.0:
        if regularizer == "l1":
            x = layers.Dense(perceptrons, activation = activation, kernel_regularizer = regularizers.l1(reg_value))(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(dropout_rate)(x)
            x = layers.Dense(perceptrons, kernel_regularizer = regularizers.l1(reg_value))(x)
            x = layers.BatchNormalization()(x)
        elif regularizer == "l2":
            x = layers.Dense(perceptrons, activation = activation, kernel_regularizer = regularizers.l2(reg_value))(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(dropout_rate)(x)
            x = layers.Dense(perceptrons, kernel_regularizer = regularizers.l2(reg_value))(x)
            x = layers.BatchNormalization()(x)
            
    else: 
        if regularizer == "l1":
            x = layers.Dense(perceptrons, activation = activation, kernel_regularizer = regularizers.l1(reg_value))(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dense(perceptrons, kernel_regularizer = regularizers.l1(reg_value))(x)
            x = layers.BatchNormalization()(x)
        elif regularizer == "l2":
            x = layers.Dense(perceptrons, activation = activation, kernel_regularizer = regularizers.l2(reg_value))(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dense(perceptrons, kernel_regularizer = regularizers.l2(reg_value))(x)
            x = layers.BatchNormalization()(x)

    residual = layers.Dense(perceptrons)(residual)

    x = layers.add([x, residual]) 
    x = layers.Activation(activation)(x)
        
    return x