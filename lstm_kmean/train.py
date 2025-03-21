import tensorflow as tf
import numpy as np
from glob import glob
from natsort import natsorted
import os
import pickle
from model import TripleNet, train_step, test_step
from utils import load_complete_data
from tqdm import tqdm
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns
import pandas as pd
from sklearn.cluster import KMeans

# Prevents duplicate library issues



# ✅ Force TensorFlow to use GPU if available
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)  # Prevents GPU memory overflow
        tf.config.set_visible_devices(gpus[0], 'GPU')  # Use the first available GPU
        print("Using GPU:", gpus[0])
    except RuntimeError as e:
        print("Error enabling GPU:", e)
else:
    print("No GPU found, training will run on CPU.")

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

#Fix random seed for reproducibility
np.random.seed(45)
tf.random.set_seed(45)

if __name__ == '__main__':
    n_channels  = 14
    n_feat      = 128
    batch_size  = 256
    test_batch_size  = 1
    n_classes   = 10

    # Load EEG Dataset
    with open('../data/eeg/image/data.pkl', 'rb') as file:
        data = pickle.load(file, encoding='latin1')
        train_X = data['x_train']
        train_Y = data['y_train']
        test_X = data['x_test']
        test_Y = data['y_test']

    # Create Batches
    train_batch = load_complete_data(train_X, train_Y, batch_size=batch_size)
    val_batch   = load_complete_data(test_X, test_Y, batch_size=batch_size)
    test_batch  = load_complete_data(test_X, test_Y, batch_size=test_batch_size)
    X, Y = next(iter(train_batch))

    # Initialize LSTM Model
    triplenet = TripleNet(n_classes=n_classes)
    opt = tf.keras.optimizers.Adam(learning_rate=3e-4)

    # Load Checkpoint if Exists
    triplenet_ckpt = tf.train.Checkpoint(step=tf.Variable(1), model=triplenet, optimizer=opt)
    triplenet_ckptman = tf.train.CheckpointManager(triplenet_ckpt, directory='experiments/best_ckpt', max_to_keep=5000)
    triplenet_ckpt.restore(triplenet_ckptman.latest_checkpoint)
    START = int(triplenet_ckpt.step) // len(train_batch)

    if triplenet_ckptman.latest_checkpoint:
        print('Restored from the latest checkpoint, epoch:', START)

    # Training Parameters
    EPOCHS = 3000
    cfreq  = 10  # Checkpoint frequency

    # Training Loop
    for epoch in range(START, EPOCHS):
        train_loss = tf.keras.metrics.Mean()
        test_loss  = tf.keras.metrics.Mean()

        # Train LSTM
        tq = tqdm(train_batch)
        for idx, (X, Y) in enumerate(tq, start=1):
            loss = train_step(triplenet, opt, X, Y)
            train_loss.update_state(loss)
            tq.set_description(f'Train Epoch: {epoch}, Loss: {train_loss.result():.4f}')

        # Validate LSTM
        tq = tqdm(val_batch)
        for idx, (X, Y) in enumerate(tq, start=1):
            loss = test_step(triplenet, X, Y)
            test_loss.update_state(loss)
            tq.set_description(f'Validation Epoch: {epoch}, Loss: {test_loss.result():.4f}')

        # Save Checkpoints
        triplenet_ckpt.step.assign_add(1)
        if (epoch % cfreq) == 0:
            triplenet_ckptman.save()

    # Extract & Save Latent Representations
    os.makedirs("saved_features", exist_ok=True)  # Create directory if it doesn't exist
    print("Extracting EEG Latent Representations...")

    latent_X = []
    latent_Y = []
    tq = tqdm(test_batch)
    for idx, (X, Y) in enumerate(tq, start=1):
        Y_emb, _ = triplenet(X, training=False)  # Get LSTM output
        latent_X.extend(Y_emb.numpy())  # Store features
        latent_Y.extend(Y.numpy())  # Store labels

    latent_X = np.array(latent_X)
    latent_Y = np.array(latent_Y)
    print('Latent Y shape: ', latent_Y.shape)

    # Save Features and Labels
    np.save("saved_features/latent_features.npy", latent_X)  
    with open("saved_features/latent_features.pkl", "wb") as f:
        pickle.dump(latent_X, f)

    np.save("saved_features/latent_labels.npy", latent_Y)  
    with open("saved_features/latent_labels.pkl", "wb") as f:
        pickle.dump(latent_Y, f)

    print("Latent Representations Saved to 'saved_features/'")

