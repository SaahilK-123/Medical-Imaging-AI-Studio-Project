# INITIALISATION TEST <PASSED>

# import tensorflow as tf
# print("TensorFlow version:", tf.__version__)

# import pandas as pd
# print("Pandas version:", pd.__version__)

# pip show ____ --version (fill in ____ with the name of the package you wish to find) [put these in the terminal line]

# AI Scripting Begins Here:

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import callbacks
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.models import Model
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

from MedData_Loader import load_images

# The "<FILE PATH>" variable should be replaced with the File Path String extracted from "right-clicking" the folder 
# I.e.: "C:/Desktop/User/Project_Name"
# If you do not want this variable to be exposed you can instead place this in a separate (.env) file

IMG_DIR = r"<FILE PATH>"

# Threshold is the split for Training-Testing
# gray=True / gray=False is "toggle" for Black and White vs. Colour

images, masks = load_images(IMG_DIR, gray=True, target_size=(128, 128), threshold=0.3)

def resize(input_image, input_mask):
    
    input_image = tf.image.resize(input_image, (128, 128), method='nearest')
    input_mask = tf.image.resize(input_mask, (128, 128), method='nearest')
    
    return input_image, input_mask

def augment(input_image, input_mask):
    
    if tf.random.uniform(()) > 0.5:

        input_image = tf.image.flip_left_right(input_image)
        input_mask = tf.image.flip_left_right(input_mask)

    return input_image, input_mask

def normalise(input_image, input_mask):

    input_image = tf.cast(input_image, tf.float32) / 255.0
    input_mask -= 1
    
    return input_image, input_mask

def load_image_train(datapoint):
    
    input_image = datapoint["image"]
    input_mask = datapoint["segmentation_mask"]
    input_image, input_mask = resize(input_image, input_mask)
    input_image, input_mask = augment(input_image, input_mask)
    input_image, input_mask = normalise(input_image, input_mask)

    return input_image, input_mask

def load_image_test(datapoint):
    
    input_image = datapoint["image"]
    input_mask = datapoint["segmentation_mask"]
    input_image, input_mask = resize(input_image, input_mask)
    input_image, input_mask = normalise(input_image, input_mask)

    return input_image, input_mask

X_train, X_test, y_train, y_test = train_test_split(images, masks, test_size=0.2, random_state=42)

train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).prefetch(tf.data.AUTOTUNE)
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).prefetch(tf.data.AUTOTUNE)

# DATA PARSING TEST <PASSED>
# print(train_dataset)

BATCH_SIZE = 8
BUFFER_SIZE = 100

train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_batches = train_dataset.cache().shuffle(100).batch(BATCH_SIZE).repeat().prefetch(tf.data.experimental.AUTOTUNE)

val_dataset = tf.data.Dataset.from_tensor_slices((X_test[:10], y_test[:10]))
val_batches = val_dataset.batch(BATCH_SIZE)

test_dataset = tf.data.Dataset.from_tensor_slices((X_test[:10], y_test[:10]))
test_batches = test_dataset.batch(BATCH_SIZE)

# train_batches = train_dataset.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
# train_batches = train_batches.prefetch(buffer_size = tf.data.experimental.AUTOTUNE)
# validation_batches = test_dataset.take(20).batch(BATCH_SIZE)
# test_batches = test_dataset.batch(BATCH_SIZE)

def display(display_list):
    plt.figure(figsize=(15,15)) # figsize values can be changed (if need be)

    title = ["Input Image", "True Mask", "Predicted Mask"]

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.utils.array_to_img(display_list[i]))
        plt.axis("off")
    plt.show()

sample_batch = next(iter(train_batches))
random_index = np.random.choice(sample_batch[0].shape[0])

# Changing numeric value in .take() changes the input image => 1 gives best case for demonstration
for image, mask in train_dataset.take(1): 
    sample_image, sample_mask = image, mask

# DATA MASKING TEST <PASSED>
# display([sample_image, sample_mask])

def double_conv_block(x, n_filters):
    
    x = layers.Conv2D(n_filters, 3, padding="same", activation="relu", kernel_initializer="he_normal")(x)
    x = layers.Conv2D(n_filters, 3, padding="same", activation="relu", kernel_initializer="he_normal")(x)
    
    return x

def downsample_block(x, n_filters):
    
    f = double_conv_block(x, n_filters)
    p = layers.MaxPool2D(2)(f)
    p = layers.Dropout(0.3)(p)

    return f, p

def upsample_block(x, conv_features, n_filters):

    x = layers.Conv2DTranspose(n_filters, 3, strides=2, padding="same")(x)
    x = layers.concatenate([x, conv_features])
    x = layers.Dropout(0.3)(x)
    x = double_conv_block(x, n_filters)

    return x

def build_unet_model():

    inputs = layers.Input(shape=(128,128,1))

    f1, p1 = downsample_block(inputs, 64)
    f2, p2 = downsample_block(p1, 128)
    f3, p3 = downsample_block(p2, 256)
    f4, p4 = downsample_block(p3, 512)

    bottleneck = double_conv_block(p4, 1024)

    u6 = upsample_block(bottleneck, f4, 512)
    u7 = upsample_block(u6, f3, 256)
    u8 = upsample_block(u7, f2, 128)
    u9 = upsample_block(u8, f1, 64)

    outputs = layers.Conv2D(3, 1, padding="same", activation="softmax")(u9)

    unet_model = tf.keras.Model(inputs, outputs, name="U-Net")

    return unet_model

unet_model = build_unet_model()

# unet_model.compile(optimizer=keras.optimizers.Adam(), loss="sparse_categorical_crossentropy", metrics=["accuracy"])

def dice_loss(y_true, y_pred, smooth=1e-6):
    y_true = tf.one_hot(tf.squeeze(tf.cast(y_true, tf.int32), axis=-1), depth=tf.shape(y_pred)[-1])
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)

    intersection = tf.reduce_sum(y_true * y_pred, axis=[1,2])
    union = tf.reduce_sum(y_true + y_pred, axis=[1,2])
    dice = (2 * intersection +smooth) / (union + smooth)

    return 1 - tf.reduce_mean(dice)

sparce_ce = SparseCategoricalCrossentropy()

def combo_loss(y_true, y_pred):
    ce_loss = sparce_ce(y_true, y_pred)
    d_loss = dice_loss(y_true, y_pred)
    
    return ce_loss + d_loss

unet_model.compile(optimizer=keras.optimizers.Adam(), loss=combo_loss, metrics=["accuracy"])

NUM_EPOCHS = 20

TRAIN_LENGTH = len(X_train)
STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE

VAL_SUBSPLITS = 5
TEST_LENGTH = len(X_test)
VALIDATION_STEPS = TEST_LENGTH // BATCH_SIZE // VAL_SUBSPLITS

early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

model_history = unet_model.fit(train_batches, 
                               epochs=NUM_EPOCHS, 
                               steps_per_epoch=STEPS_PER_EPOCH,
                               validation_data=val_batches, 
                               validation_steps=VALIDATION_STEPS,
                               callbacks=[early_stopping])

def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    
    return pred_mask

def display_sample(image, true_mask, pred_mask):
    
        plt.figure(figsize=(12,4))

        plt.subplot(1, 3, 1)
        plt.title("Input Image")
        plt.imshow(tf.squeeze(image), cmap="gray")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.title("True Mask")
        plt.imshow(tf.squeeze(true_mask), cmap="gray")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.title("Predicted Mask")
        plt.imshow(tf.squeeze(pred_mask), cmap="gray")
        plt.axis("off")

        plt.tight_layout()
        plt.show()

def show_predictions(data=None, num=1):
    if data:
        for images, masks in data.take(num):
            pred_masks =  unet_model.predict(images)
            pred_masks = create_mask(pred_masks)

            for i in range(min(3, len(images))):
                display_sample(images[i], masks[i], pred_masks[i])
    else:
        pred_masks = unet_model.predict(sample_image[tf.newaxis, ...])
        display_sample(sample_image, sample_mask, create_mask(pred_masks))

count = 0
for i in test_batches:
    count += 1
print("number of batches", count)

show_predictions(data=test_batches, num=3)

for images, masks in test_batches.take(1):
    pred_masks = unet_model.predict(images)
    pred_masks = create_mask(pred_masks)

    for i in range(min(3, len(images))):
        display_sample(images[i], masks[i], pred_masks[i])

final_accuracy = model_history.history.get("accuracy", [None])[-1]
final_loss = model_history.history.get("loss", [None])[-1]
print(f"Final Training Accuracy: {final_accuracy:.4f}")
print(f"Final Training Loss: {final_loss:.4f}")

def plot_training_history(history):
    acc = history.history.get('accuracy', [])
    val_acc = history.history.get('val_accuracy', [])
    loss = history.history.get('loss', [])
    val_loss = history.history.get('val_loss', [])
    epochs = range(len(acc))

    if not val_acc or not val_loss:
        print("Validation metrics are missing. Ensure validation_data and validation_steps are set correctly.")
        return

    # Graph 1: Accuracy
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, acc, label='Training Accuracy')
    plt.plot(epochs, val_acc, label='Validation Accuracy')
    plt.title('Training vs Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Graph 2: Loss
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, loss, label='Training Loss')
    plt.plot(epochs, val_loss, label='Validation Loss')
    plt.title('Training vs Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

plot_training_history(model_history)

count = 0
for i in test_batches:
    count += 1
print("number of batches", count)

show_predictions(data=test_batches, num=3)

def mask_comaprison(image, true_mask, pred_mask, index=0):
    
    plt.figure(figsize=(8,4))

    plt.subplot(1, 2, 1)
    plt.title("True Mask")
    plt.imshow(tf.squeeze(true_mask), cmap="gray")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    plt.subplot(1, 2, 2)
    plt.title("Predicted Mask")
    plt.imshow(tf.squeeze(pred_mask), cmap="gray")
    plt.axis("off")
    plt.tight_layout()
    plt.show()
    
model.save("MedAI_Results.keras")