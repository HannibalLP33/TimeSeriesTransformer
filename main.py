import pandas as pd
import numpy as np
import logging
import tensorflow as tf
from tensorflow.keras import layers, Input, Model
from tensorflow.keras.layers import Lambda, Reshape, Concatenate, Add, Dense, Dropout, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D, Layer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score

logging.basicConfig(filename = "logs.log", format = "%(asctime)s -- %(message)s", datefmt='%m/%d/%Y %I:%M:%S %p', level = logging.INFO)
training_data = pd.read_csv("training_data/Attempt2.csv")
training_data.head()
X = training_data.iloc[:,:10]
y = training_data.iloc[:,10]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, shuffle = False)

logging.info("Dataset Upload successfully")
logging.info(f"X training set shape:{X_train.shape}")
logging.info(f"y training set shape:{y_train.shape}")
logging.info(f"X test set shape:{X_test.shape}")
logging.info(f"y test set shape:{y_test.shape}")


print("Dataset Upload successfully")
print(f"X training set shape:{X_train.shape}")
print(f"y training set shape:{y_train.shape}")
print(f"X test set shape:{X_test.shape}")
print(f"y test set shape:{y_test.shape}")

sequence_len = 10 # Looking at 24 hours worth of data to determine
epochs = 5
attention_heads = 6
projection_dim = sequence_len
dropout = 0.1
num_transformer_blocks = 8
mlp_units = [2048, 1024, 500, 250, 100, 10]
tranformer_mlp_units = [projection_dim ** 2, projection_dim * 2]
X_train = tf.expand_dims(X_train, axis = 1)


class Time2Vector(tf.keras.layers.Layer):
    def __init__(self, kernal: int = 64):
        super().__init__(trainable = True,  name = "Time2VecLayer")
        self.kernal = kernal - 1
        
    def build(self, input_shape):
        ### Time to Vector Piecewise function ###
        
        self.weights_linear = self.add_weight(shape = (input_shape[1],1), initializer = "uniform", trainable = True, name = "weights_linear")
        
        self.bias_linear = self.add_weight(shape = (input_shape[1],1), initializer = "uniform", trainable = True, name = "bias_linear")
        
        self.weights_periodic = self.add_weight(shape = (input_shape[1], self.kernal), initializer = "uniform", trainable = True, name = "weights_periodic")
        
        self.bias_periodic = self.add_weight(shape = (input_shape[1], self.kernal), initializer = "uniform", trainable = True, name = "bias_periodic")
    
    def call(self, x):
        x = tf.expand_dims(x[:,:,0], axis = -1)
        time_linear = (self.weights_linear * x) + self.bias_linear
        # time_linear = tf.expand_dims(time_linear, axis = -1) #Expand dimensions to concat later
        
        time_periodic = tf.math.sin(tf.multiply(x, self.weights_periodic) + self.bias_periodic)
        # time_periodic = tf.expand_dims(time_periodic, axis = -1)
        
        final_product = tf.concat([time_linear, time_periodic], axis = -1)
        # final_product = tf.expand_dims(final_product, axis = 0)
        return final_product
    
def mlp_block(x, units):
    x = GlobalAveragePooling1D(data_format = "channels_first")(x)
    for unit in units:
        x = Dense(unit, activation = tf.nn.gelu)(x)
        x = Dropout(0.1)(x)
    return x
def transformer_encoder(inputs, attention_heads, projection_dim, dropout):
    ### Layer Normalization / Multihead Attention Layers ###
    x = LayerNormalization(epsilon = 1e-6)(inputs)
    x = MultiHeadAttention(num_heads = attention_heads, key_dim = projection_dim, dropout = dropout)(x,x)
    skip1 = Add()([x, inputs])
    
    ### Feed Forward ###
    x = LayerNormalization(epsilon = 1e-6)(skip1)
    x = mlp_block(x, tranformer_mlp_units)
    skip2 = Add()([x,skip1])
    
    return skip2


def build_model():
    input = Input(shape = X_train.shape[1:]) # (Batch_size, Sequence Length, Number of Features)
    x = Time2Vector(sequence_len)(input)
    x = Concatenate(axis = -1)([input, x]) 
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, attention_heads, projection_dim, dropout)
    x = mlp_block(x, mlp_units)
    output = Dense(1, activation = "relu")(x)
    
    model = Model(inputs = input, outputs = output)
    model.summary()

    return model 

def train_model(model):
    optimizer = tf.optimizers.Adam(learning_rate=1e-3, decay = 1e-4)
    checkpoint_path = "/models/"
    model.compile(optimizer=optimizer, 
                  loss = tf.keras.losses.MeanAbsoluteError(), 
                  metrics = [tf.keras.metrics.MeanSquaredError(name = "MSE"), 
                             tf.keras.metrics.RootMeanSquaredError(name = "RMSE"),
                             tf.keras.metrics.MeanAbsoluteError(name = "MAE")])
    
    history = model.fit(
        x = X_train,
        y = y_train,
        epochs = epochs,
        batch_size = X_train.shape[0],
        validation_split = 0.2
        # callbacks = [checkpoint_callback],
        
    )
    
    return history
 
def main ():
    model = build_model()
    training = train_model(model)

if __name__ == "__main__":
    main()