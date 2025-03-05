import gc
import os
import csv
import wandb
import joblib
import numpy as np
import tensorflow as tf
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping
from code.model_train.resunet import resnet_attention

def setup_gpu():
    """
    Configure the environment for GPU usage and enable dynamic memory allocation.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"  # Specify available GPUs
    os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'  # Enable asynchronous memory allocation

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            print(e)

def custom_loss(y_true, y_pred, alpha=0.5, beta=0.5):
    """
    Custom loss function combining weighted MAE and max value difference.
    
    Args:
        y_true: Ground truth tensor.
        y_pred: Predicted tensor.
        alpha: Weight for MAE loss component.
        beta: Weight for max value difference loss component.
    
    Returns:
        Total loss combining weighted MAE and max value difference loss.
    """
    weight = (1 / (110.0**2)) * (tf.abs(y_true - 68)**2) + 1.0  # Compute weight based on deviation
    mae_loss = tf.reduce_mean(tf.abs(y_true - y_pred) * weight)
    
    max_true = tf.reduce_max(y_true, axis=1)
    max_pred = tf.reduce_max(y_pred, axis=1)
    max_diff_loss = tf.reduce_mean(tf.abs(max_true - max_pred))
    
    return alpha * mae_loss + beta * max_diff_loss

def compile_model(strategy, steps, learning_rate=5e-4, nibp=True):
    """
    Compile the ResUNet model with a specified training strategy and learning rate schedule.
    
    Args:
        strategy: TensorFlow distribution strategy for multi-GPU training.
        steps: Number of training steps per epoch.
        learning_rate: Initial learning rate.
        nibp: Boolean flag to include NIBP in model.
    
    Returns:
        Compiled Keras model.
    """
    with strategy.scope():
        model = resnet_attention(length=1024, filter=32, depth=3, nibp=nibp)
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            learning_rate, decay_steps=steps, decay_rate=0.96, staircase=True)
        
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule, epsilon=1e-4),
                      loss=custom_loss, metrics=['mae'])
    return model

class LrTracker(Callback):
    """
    Callback to log learning rate and save model checkpoints during training.
    """
    def __init__(self, model_save_path):
        self.model_save_path = model_save_path

    def on_train_begin(self, logs=None):
        print("Training has started...")

    def _get_current_lr(self):
        lr = self.model.optimizer.lr
        if isinstance(lr, tf.keras.optimizers.schedules.LearningRateSchedule):
            lr = lr(self.model.optimizer.iterations)
        return tf.keras.backend.get_value(lr)

    def on_epoch_begin(self, epoch, logs=None):
        print(f'Epoch {epoch+1}: Learning rate is {self._get_current_lr():.7f}.')

    def on_epoch_end(self, epoch, logs=None):
        with open(os.path.join(self.model_save_path, 'train_log.csv'), 'a', newline='', encoding='cp949') as f:
            wr = csv.writer(f)
            wr.writerow([epoch+1, logs.get('loss'), logs.get('val_loss'), self._get_current_lr()])
        
        if epoch % 5 == 0:
            checkpoint_filepath = os.path.join(self.model_save_path, f"checkpoint/checkpoint_epoch_{epoch+1}_valloss_{logs.get('val_loss', 0):.4f}_lr_{self._get_current_lr():.7f}.h5")
            self.model.save(checkpoint_filepath)
        
        wandb.log({'epoch': epoch+1, 'loss': logs.get('loss'), 'mae': logs.get('mae'),
                   'val_loss': logs.get('val_loss'), 'val_mae': logs.get('val_mae'),
                   'learning_rate': self._get_current_lr()})

def train_model(model, train_dataset, val_dataset, steps, val_steps, model_save_path, checkpoint_path):
    """
    Train the model using the specified datasets, steps, and checkpoints.
    
    Args:
        model: Compiled Keras model.
        train_dataset: TensorFlow dataset for training.
        val_dataset: TensorFlow dataset for validation.
        steps: Number of training steps per epoch.
        val_steps: Number of validation steps per epoch.
        model_save_path: Directory to save model logs and checkpoints.
        checkpoint_path: Path to save the best model checkpoint.
    
    Returns:
        Training history object.
    """
    checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True, mode='auto', save_weights_only=False, save_freq='epoch')
    early_stop = EarlyStopping(monitor='val_loss', mode='min', patience=25)
    
    gc.collect()
    
    history = model.fit(train_dataset, epochs=200, steps_per_epoch=steps,
                        validation_data=val_dataset, validation_steps=val_steps,
                        callbacks=[early_stop, checkpoint, LrTracker(model_save_path)])
    return history

def evaluate_model(model, test_data, batch_size, model_save_path):
    """
    Evaluate the model on the test set and save predictions.
    
    Args:
        model: Trained Keras model.
        test_data: Tuple containing test dataset (y, ppg, nibp_s, nibp_m, nibp_d, nibp_a).
        batch_size: Batch size for prediction.
        model_save_path: Path to save evaluation results.
    
    Returns:
        Mean Absolute Error (MAE) of model predictions.
    """
    test_Y, test_PPG, test_NIBP_S, test_NIBP_M, test_NIBP_D, test_NIBP_A = test_data
    predictions = model.predict([test_PPG, test_NIBP_S, test_NIBP_M, test_NIBP_D, test_NIBP_A], batch_size=batch_size, verbose=1)
    mae = np.mean(np.abs(test_Y - predictions))
    joblib.dump((test_Y, predictions), os.path.join(model_save_path, 'test_files.pkl'), compress=3)
    return mae
