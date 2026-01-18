import tensorflow as tf
import logging
from config import L2_REGULARIZATION, DROPOUT_RATE, LEARNING_RATE


def create_model(input_shape):
    """
    A feed-forward regression model.

    Notes:
    - Input is a fixed-size feature vector (look-back window after preprocessing).
    - L2 regularization + dropout help reduce overfitting.
    - Output is a single number (next-step consumption).
    """
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(input_shape,)),

            # Bigger layer first: learns a strong representation
            tf.keras.layers.Dense(
                512,
                activation="relu",
                kernel_regularizer=tf.keras.regularizers.l2(L2_REGULARIZATION),
            ),
            tf.keras.layers.Dropout(DROPOUT_RATE),

            # Gradually reduce width to compress useful features
            tf.keras.layers.Dense(
                256,
                activation="relu",
                kernel_regularizer=tf.keras.regularizers.l2(L2_REGULARIZATION),
            ),
            tf.keras.layers.Dropout(DROPOUT_RATE),

            tf.keras.layers.Dense(
                128,
                activation="relu",
                kernel_regularizer=tf.keras.regularizers.l2(L2_REGULARIZATION),
            ),

            # Regression output: no activation
            tf.keras.layers.Dense(1),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="mean_squared_error",
        metrics=["mae"],
    )

    logging.info("Model created and compiled.")
    return model
