import numpy as np
import glob
import os, sys
import tensorflow as tf
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
here = os.path.dirname(__file__)
parent = os.path.abspath(os.path.join(here, ".."))
if parent not in sys.path:
    sys.path.insert(0, parent)
from utils import PRED_TO_SYMBOL
from tqdm.keras import TqdmCallback
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def load_dataset(file_base_path):
    # ==========================================
    # 1. DEFINITION OF CLASSES
    # ==========================================
    # WARNING: Ensure the numeric order (0,1,2,3)
    # matches the one used by the event-based-roshambo-demo app!
    class_mapping = PRED_TO_SYMBOL
    
    X_list = []
    Y_list = []
    
    # ==========================================
    # 2. READ FILES
    # ==========================================
    print("Beginning loading data...")
    
    for id_class, name_class in class_mapping.items():
        # Build the path, e.g. "my_numpy_dataset/rock/*.npy"
        dataset_folder_path = os.path.join(file_base_path, f"{name_class}*.npy")
        print(dataset_folder_path)
        found_files = glob.glob(dataset_folder_path)
        
        total_frames_per_class = 0
        
        for file in found_files:
            # Load frames (expected shape: N, 64, 64, 1)
            frames = np.load(file)
            X_list.append(frames)
            
            # Create an array of identical labels for this block
            # e.g. if there are 500 frames, create an array of 500 zeros
            labels_array = np.full(len(frames), id_class)
            Y_list.append(labels_array)
            
            total_frames_per_class += len(frames)

            
        print(f"Class '{name_class}': loaded {len(found_files)} files for a total of {total_frames_per_class} frames.")

    if not X_list:
        raise ValueError("No .npy files found. Check the folder paths!")

    # ==========================================
    # 3. CONCATENATION, ENCODING AND SHUFFLING
    # ==========================================
    # Concatenate all lists into two big NumPy arrays
    print('Concateneting lists...')
    X_all = np.concatenate(X_list, axis=0)
    Y_all = np.concatenate(Y_list, axis=0)
    
    del X_list  # Free memory by deleting the original lists
    del Y_list

    print(f"\nTotal combined dataset shape: {X_all.shape}")

    # One-Hot Encoding: transforms label '2' into [0, 0, 1, 0]
    Y_all = tf.keras.utils.to_categorical(Y_all, num_classes=len(class_mapping))

    
    print("Data labeled and shuffled successfully!")
    
    return X_all, Y_all

def fine_tune(X, y):

    print('Starting fine tuning...')
    X, y = shuffle(X, y, random_state=42)
    # Decide how many original frames to keep (e.g. 40,000 total)
    NUM_SAMPLES_TO_KEEP = 100000

    X_subset = X[:NUM_SAMPLES_TO_KEEP]
    y_subset = y[:NUM_SAMPLES_TO_KEEP]

    # Free memory IMMEDIATELY by deleting the large arrays
    del X
    del y

    print(f"Subset extracted for fine-tuning: {X_subset.shape}")

    # 4. Now generate rotations ONLY on this small subset
    print("Generating rotations...")
    X_90  = np.rot90(X_subset, k=1, axes=(1, 2))
    X_270 = np.rot90(X_subset, k=3, axes=(1, 2))

    # Labels do not change with rotation!
    y_90  = y_subset.copy()
    y_270 = y_subset.copy()

    # 5. Combine the three small datasets
    # We'll have 20k original + 20k at 90° + 20k at 270° = 60,000 frames for fine-tuning
    X_finetune = np.concatenate([X_subset, X_90, X_270], axis=0)
    y_finetune = np.concatenate([y_subset, y_90, y_270], axis=0)

    # Final shuffle to mix original and rotated samples during training
    X_finetune, y_finetune = shuffle(X_finetune, y_finetune, random_state=42)
        
    X_train, X_val, y_train, y_val = train_test_split(X_finetune, y_finetune, test_size=0.2, random_state=42)
    
    print("\nReady for training:")
    print(f"X_train: {X_train.shape} | Y_train: {y_train.shape}")
    
    # ESEMPIO: Qui si collega il codice di prima!
    model = tf.keras.models.load_model('model/roshambo.h5')
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    # Freeze all layers except the last 2 or 3
    for layer in model.layers[:-3]:
        layer.trainable = False

    # Compila il modello
    model.compile(
        optimizer=optimizer, 
        loss='categorical_crossentropy', 
        metrics=['accuracy', 'categorical_crossentropy']
        )

    # Optional: stop training if validation loss doesn't improve for 3 epochs
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=3, 
        restore_best_weights=True
        )
    
    datagen = ImageDataGenerator(
        rotation_range=15,      # +/- 15 degrees: teaches it that "vertical" can be a bit tilted
        width_shift_range=0.1,  # User's hand isn't perfectly centered
        height_shift_range=0.1, # User's hand is too high/low
        zoom_range=0.1          # User is slightly closer/further from the camera
    )

    print("Starting fine-tuning...")

    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=32),
        validation_data=(X_val, y_val),
        epochs=15,          
        batch_size=32,      
        verbose=0,          
        callbacks=[early_stopping, TqdmCallback(verbose=1)]
    )

    final_train_loss = history.history['loss'][-1]
    final_val_loss = history.history['val_loss'][-1]
    eval_results = model.evaluate(X_val, y_val, verbose=0, return_dict=True)

    print("\nFinal training cross-entropy:", f"{final_train_loss:.4f}")
    print("Final validation cross-entropy:", f"{final_val_loss:.4f}")
    print("Validation cross-entropy after restoring best weights:", f"{eval_results['loss']:.4f}")
    if 'accuracy' in eval_results:
        print("Validation accuracy after restoring best weights:", f"{eval_results['accuracy']:.4f}")

    # ==========================================
    # 5. SAVING AND CONVERSION TO TFLITE
    # ==========================================

    # Save the new Keras model
    model.save('model/dextra_roshambo_finetuned_new.h5')
    print("Keras model saved successfully!")

    # Convert the new model to TFLite for the Demo
    print("Converting to TensorFlow Lite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    with open('model/finetuned_model_dextra_roshambo_new.tflite', 'wb') as f:
        f.write(tflite_model)

    print("Conversion completed! Replace the file 'finetuned_model_dextra_roshambo.tflite' in your app.")


# ==========================================
# TEST AND FINE-TUNING PREPARATION
# ==========================================
if __name__ == "__main__":
    # Replace with the path to the main folder that contains the 4 subfolders
    DATASET_DIRECTORY = "..\\aedat_files" 
    
    X, Y = load_dataset(DATASET_DIRECTORY)
    fine_tune(X, Y)
