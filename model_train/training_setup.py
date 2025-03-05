import gc
import os
import csv
import wandb
import joblib
import numpy as np
import tensorflow as tf
from keras.callbacks import Callback
from keras.callbacks import ModelCheckpoint, EarlyStopping
from code.model_train.resunet import resnet_attention


def setup_gpu():
    # Configure the environment for GPU usage and set up GPU memory management.
    os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3" # Specify the GPUs to be used
    os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async' # Use the async memory allocator for TensorFlow

    gpus = tf.config.experimental.list_physical_devices('GPU') # List the available physical GPUs
    if gpus:
        try:
            for gpu in gpus:  # Enable dynamic memory allocation on each GPU
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            print(e)


def custom_loss(y_true, y_pred, alpha=0.5, beta=0.5):
    """
    Custom Loss Function that combines weighted MAE and max value difference.

    Args:
        y_true: Ground truth tensor.
        y_pred: Predicted tensor.
        alpha: Weight for the weighted MAE component.
        beta: Weight for the max value difference component.

    Returns:
        Combined loss value.
    """
    # Weighted Mean Absolute Error (MAE)
    weight = (1 / (110.0**2)) * (tf.abs(y_true - 68)**2) + 1.0  # Calculate weight based on target deviation
    mae_loss = tf.reduce_mean(tf.abs(y_true - y_pred) * weight)

    # Max Value Difference Loss
    max_true = tf.reduce_max(y_true, axis=1)  # Batch-wise max of y_true
    max_pred = tf.reduce_max(y_pred, axis=1)  # Batch-wise max of y_pred
    max_diff_loss = tf.reduce_mean(tf.abs(max_true - max_pred))  # Mean absolute difference of max values

    # Combined Loss
    total_loss = alpha * mae_loss + beta * max_diff_loss
    return total_loss


def compile_model(strategy, steps, learning_rate=5e-4, nibp=True):
    # Compile the model with the specified strategy, learning rate schedule, and custom loss function.
    with strategy.scope():
        # Initialize the model using the ResNet scalar architecture
        model = resnet_attention(length=1024, filter=32, depth=3, nibp=nibp)
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            learning_rate,
            decay_steps=steps,
            decay_rate=0.96, 
            staircase=True)
        
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule, epsilon=1e-4), 
                      loss=custom_loss, 
                      metrics=['mae'])
    return model


class LrTracker(Callback):
    def __init__(self, model_save_path):
        # Initialize the callback with the model save path for logging purposes.
        self.model_save_path = model_save_path

    def on_train_begin(self, logs=None):
        print("Training has started...")

    def _get_current_lr(self):
        # Retrieve the current learning rate from the model's optimizer.
        lr = self.model.optimizer.lr
        if isinstance(lr, tf.keras.optimizers.schedules.LearningRateSchedule):
            lr = lr(self.model.optimizer.iterations)
        return tf.keras.backend.get_value(lr)

    def on_epoch_begin(self, epoch, logs=None):
        print(f'Epoch {epoch+1}: Learning rate is {self._get_current_lr():.7f}.')

    def on_epoch_end(self, epoch, logs=None):
        with open(self.model_save_path + '/train_log.csv', 'a', newline='', encoding='cp949') as f:
            wr = csv.writer(f)
            wr.writerow([epoch+1, logs.get('loss'), logs.get('val_loss'), self._get_current_lr()])

        # Save model checkpoints every 5 epochs
        if epoch % 5 == 0:
            filepath = f"{self.model_save_path}/checkpoint/checkpoint_epoch_{epoch+1}_valloss_{logs.get('val_loss', 0):.4f}_lr_{self._get_current_lr():.7f}.h5"
            self.model.save(filepath) 
        
        # Log the training and validation metrics to Weights & Biases
        wandb.log({
            'epoch': epoch+1,
            'loss': logs.get('loss'),
            'mae': logs.get('mae'),
            'val_loss': logs.get('val_loss'),
            'val_mae': logs.get('val_mae'),
            'learning_rate': self._get_current_lr()
        })


def train_model(model, train_dataset, val_dataset, steps, val_steps, model_save_path, checkpoint_path):
    # Train the model using the specified training and validation datasets, and save the best model.
    checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True, mode='auto', save_weights_only=False, save_freq='epoch')
    es = EarlyStopping(monitor='val_loss', mode='min', patience=25) 
    
    gc.collect()
    
    history = model.fit(train_dataset, 
                        epochs=200, 
                        steps_per_epoch=steps,
                        validation_data=val_dataset, 
                        validation_steps=val_steps, 
                        callbacks=[
                                es, 
                                checkpoint, 
                                LrTracker(model_save_path)
                            ]
                        ) 
    return history


def evaluate_model(model, test_data, batch_size, model_save_path):
    # Evaluate the model on the test data and calculate the Mean Absolute Error (MAE).
    # The predictions and actual values are saved for further analysis.

    # Unpack the test data into respective variables
    test_Y, test_PPG, test_NIBP_S, test_NIBP_M, test_NIBP_D, test_NIBP_A = test_data

    # Use the model to predict the outputs based on the test input data
    pred = model.predict([test_PPG, test_NIBP_S, test_NIBP_M, test_NIBP_D, test_NIBP_A], batch_size=batch_size, verbose=1) 

    # Calculate the Mean Absolute Error (MAE) between the predicted and actual values
    mae = np.mean(np.abs(test_Y - pred))

    joblib.dump((test_Y, pred), model_save_path + '/test_files.pkl', compress=3)

    return mae