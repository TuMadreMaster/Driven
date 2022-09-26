from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC
from tensorflow.keras.losses import BinaryCrossentropy
from keras import Model, Input
from keras import layers

import keras_tuner as kt

import numpy as np

def transformer_builder_input(input_shape):

    def transformer_builder(hp: kt.HyperParameters):

        # Define the model input
        inputs: np.array = Input(shape=input_shape)
        x = inputs

        # transformer layer
        def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):

            # Normalization and Attention
            x = layers.LayerNormalization(epsilon=1e-6)(inputs)
            x = layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
            x = layers.Dropout(dropout)(x)
            res = x + inputs

            # Feed Forward Part
            x = layers.LayerNormalization(epsilon=1e-6)(res)
            x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
            x = layers.Dropout(dropout)(x)
            x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
            
            return x + res

        # Tune the number of transformer blocks (num_transformer_blocks=8)
        hp_num_transformer_blocks = hp.Int('num_transformer_blocks', min_value=1, max_value=8, step=1)

        # Tune the attention head size (head_size = 256)
        hp_head_size = hp.Choice("head_size", [16, 32, 64, 128, 256])

        # Tune the number of attention heads (num_heads = 4)
        hp_num_heads = hp.Int('num_heads', min_value=1, max_value=8, step=1)

        # Tune the transformer block feed-forward dimensions (ff_dim = 4)
        hp_ff_dim = hp.Int('ff_dim', min_value=1, max_value=8, step=1)

        # Tune the transformer dropout (dropout = 0.25)
        hp_dropout = hp.Float('dropout', min_value=0.1, max_value=1)

        for _ in range(hp_num_transformer_blocks):
            x = transformer_encoder(x, hp_head_size, hp_num_heads, hp_ff_dim, hp_dropout)

        # Tune the number of mlp units (mlp_units = [128])
        hp_mlp_unit_num = hp.Int('mlp_unit_num', min_value=1, max_value=4, step=1)
        hp_mlp_unit_size = hp.Choice("mlp_unit_size", [32, 64, 128, 256, 512])

        # Tune the mlp droupout
        hp_mlp_dropout = hp.Float('mlp_dropout', min_value=0.1, max_value=1)

        x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
        for _ in range(hp_mlp_unit_num):
            x = layers.Dense(hp_mlp_unit_size, activation="relu")(x)
            x = layers.Dropout(hp_mlp_dropout)(x)
        outputs = layers.Dense(1, activation="sigmoid")(x)

        model = Model(inputs, outputs)
        model.compile(
            loss=BinaryCrossentropy(),
            optimizer=Adam(learning_rate=1e-5),
            metrics=[
            AUC(name="roc", curve="ROC"),
            AUC(name="pr", curve="PR")    
        ]
        )

        return model

    return transformer_builder

def transformer_model(
        input_shape: tuple,
        head_size=256,
        num_heads=4,
        ff_dim=4,
        num_transformer_blocks=1,
        mlp_units=[128],
        mlp_dropout=0.4,
        dropout=0.25,
        learning_rate=1e-5 # 1e-4
    ) -> Model:

    """ Returns a compiled TNN instance. """

    inputs = Input(shape=input_shape)
    x = inputs

    # transformer layer
    def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):

        # Normalization and Attention
        x = layers.LayerNormalization(epsilon=1e-6)(inputs)
        x = layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
        x = layers.Dropout(dropout)(x)
        res = x + inputs

        # Feed Forward Part
        x = layers.LayerNormalization(epsilon=1e-6)(res)
        x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
        x = layers.Dropout(dropout)(x)
        x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
        
        return x + res
    
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = Model(inputs, outputs)

    model.compile(
        loss=BinaryCrossentropy(),
        optimizer=Adam(learning_rate=learning_rate),
        metrics=[
            AUC(name="roc", curve="ROC"),
            AUC(name="pr", curve="PR")    
        ]
    )

    return model