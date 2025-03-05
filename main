import os
import tensorflow as tf
from model_train.training_setup import compile_model, setup_gpu, train_model, evaluate_model

def train_cal_model(train_data, valid_data, test_data, model_save_path):
    """
    Train and evaluate a ResUNet-based model.
    
    Parameters:
    - train_data: tuple (y, ppg, sbp_cal, mbp_cal, dbp_cal, delta_t_cal)
    - valid_data: tuple (y, ppg, sbp_cal, mbp_cal, dbp_cal, delta_t_cal)
    - test_data: tuple (y, ppg, sbp_cal, mbp_cal, dbp_cal, delta_t_cal)
    - model_save_path: str, directory path to save the trained model
    """
    batch_size = 1024
    learning_rate = 5e-4

    # GPU setup
    setup_gpu()
    os.makedirs(model_save_path, exist_ok=True)
    best_model_path = os.path.join(model_save_path, 'best_model.h5')
    checkpoint_path = os.path.join(model_save_path, 'checkpoint')
    os.makedirs(checkpoint_path, exist_ok=True)

    print('Train & validation data loaded successfully...')

    len_trainset = len(train_data[0])
    len_valset = len(valid_data[0])

    # Multi-GPU training strategy
    strategy = tf.distribute.MirroredStrategy()

    train_dataset = (tf.data.Dataset.from_tensor_slices((train_data[1:], train_data[0]))
                     .cache().shuffle(10000).repeat()
                     .batch(batch_size // strategy.num_replicas_in_sync)
                     .prefetch(tf.data.experimental.AUTOTUNE))
    
    val_dataset = (tf.data.Dataset.from_tensor_slices((valid_data[1:], valid_data[0]))
                   .cache().repeat()
                   .batch(batch_size // strategy.num_replicas_in_sync)
                   .prefetch(tf.data.experimental.AUTOTUNE))
    
    del train_data, valid_data

    # Compile model
    model = compile_model(strategy, steps=len_trainset // batch_size, learning_rate=learning_rate, loss_type='cus_mae', nibp=True)

    # Train model
    history = train_model(model, train_dataset, val_dataset,
                          steps=len_trainset // batch_size, 
                          val_steps=len_valset // batch_size, 
                          model_save_path=model_save_path,
                          checkpoint_path=best_model_path)
    
    del train_dataset, val_dataset

    # Evaluate model
    mae = evaluate_model(model, test_data, batch_size, model_save_path)

    del model, history
    return mae


if __name__ == '__main__':
    # Users must define train_data, valid_data, test_data, and model_save_path
    train_data = None  # Replace with actual training data
    valid_data = None  # Replace with actual validation data
    test_data = None   # Replace with actual test data
    model_save_path = "./model_output"  # Change to the desired directory

    if None in (train_data, valid_data, test_data):
        raise ValueError("Please provide valid train_data, valid_data, and test_data before running.")

    mae = train_cal_model(train_data, valid_data, test_data, model_save_path)

    print(f"Mean Absolute Error (MAE) on test data: {mae:.4f}")