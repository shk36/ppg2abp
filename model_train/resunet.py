import tensorflow as tf
from code.model_train.utils import conv_block, residual_block, fc_block, positional_encoding, self_attention
import warnings
warnings.filterwarnings('ignore')


# Define the ResNet-based model architecture
def resnet_attention(
        length=None, 
        filter =None, 
        depth=None, 
        cal=None,       
        dropout=0.2):

    con_res_add = [] # List to store the outputs of the residual blocks

    # Input layer for PPG data
    inputs = tf.keras.Input(shape=(length, 1), name='ppg')

    # Block 1: Initial convolutional and residual block
    res_x = residual_block(inputs, filter, 3, strides=1, padding='same')
    conv_x = tf.keras.layers.Conv1D(filter, 3, strides=1, padding='same')(inputs)
    conv_x = conv_block(conv_x, filter, 3, strides=1, padding='same')

    conv_x = tf.keras.layers.Add()([res_x, conv_x])
    con_res_add.append(conv_x) 

    # Blocks 2 to 4: Additional convolutional and residual blocks
    for i in range(depth-1): 
        filter = filter * 2 
        res_x = residual_block(conv_x, filter, 3, strides=2, padding='same')
        conv_x = conv_block(conv_x, filter, 3, strides=2, padding='same')
        conv_x = conv_block(conv_x, filter, 3, strides=1, padding='same')

        conv_x = tf.keras.layers.Add()([res_x, conv_x])
        con_res_add.append(conv_x) 

    # Bridge block: Final convolutional and residual block before decoding
    filter = filter * 2 
    res_x = residual_block(conv_x, filter, 3, strides=2, padding='same')
    conv_x = conv_block(conv_x, filter, 3, strides=2, padding='same')
    conv_x = conv_block(conv_x, filter, 3, strides=1, padding='same')

    conv_x = tf.keras.layers.Add()([res_x, conv_x])

    conv_x = tf.keras.layers.BatchNormalization()(conv_x)
    conv_x = tf.keras.layers.Activation('relu')(conv_x)


    # If calibration inputs are used, add them to the model
    if cal:
        cal_s = tf.keras.Input(shape=(1, 1), name='cal_s') # Systolic CAL input
        cal_d = tf.keras.Input(shape=(1, 1), name='cal_d') # Diastolic CAL input
        cal_m = tf.keras.Input(shape=(1, 1), name='cal_m') # Mean CAL input
        delta_t = tf.keras.Input(shape=(1, 1), name='delta_t') # Delta T CAL input

        first_s = cal_s[:, 0]
        first_d = cal_d[:, 0]
        first_m = cal_m[:, 0]
        first_a = delta_t[:, 0]

        # Concatenate the calibration inputs and reshape them for concatenation with the main network
        cal_x = tf.keras.layers.concatenate([first_s, first_d, first_m, first_a], axis=-1)
        cal_x = tf.expand_dims(cal_x, 1)  
        cal_x = tf.tile(cal_x, [1, int(filter/2), 1])  

        # Concatenate the calibration data with the main convolutional block
        conv_x = tf.keras.layers.concatenate([conv_x, cal_x], axis=-1)

    # Additional convolutional layers after concatenation
    conv_x = tf.keras.layers.Conv1D(filter, 3, strides=1, padding='same')(conv_x)
    if cal:
        conv_x = tf.keras.layers.concatenate([conv_x, cal_x], axis=-1)
    conv_x = tf.keras.layers.BatchNormalization()(conv_x)
    conv_x = tf.keras.layers.Activation('relu')(conv_x)
    if cal:
        conv_x = tf.keras.layers.concatenate([conv_x, cal_x], axis=-1)
    conv_x = tf.keras.layers.Conv1D(filter, 3, strides=1, padding='same')(conv_x)

    con_res_add.append(conv_x) 


    # Decoding phase (upsampling and reconstruction)
    con_res_add.reverse() 


    for i in range(depth): 
        filter = filter / 2  
        conv_x = tf.keras.layers.UpSampling1D(size=2)(conv_x)
        enc_out = con_res_add[i+1]

        conv_x = tf.keras.layers.concatenate([enc_out, conv_x], axis=-1) 
        res_x = residual_block(conv_x, filter, 3, strides=1, padding='same') 
        conv_x = conv_block(conv_x, filter, 3, strides=1, padding='same')
        conv_x = conv_block(conv_x, filter, 3, strides=1, padding='same')

        conv_x = tf.keras.layers.Add()([res_x, conv_x])

    conv_x = conv_x + positional_encoding(1024, 32)
    conv_x = self_attention(conv_x, dropout)
    conv_x = fc_block(conv_x, 32, dropout)

    # Output layer
    output = tf.keras.layers.Dense(1)(conv_x) 

    # Create the model with or without calibration inputs
    if cal:
        model = tf.keras.Model(inputs = [inputs, cal_s, cal_d, cal_m, delta_t], outputs = output)

    if not cal:
        model = tf.keras.Model(inputs= inputs, outputs = output)

    return model
