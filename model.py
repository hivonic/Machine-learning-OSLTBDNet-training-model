# %% [markdown]
# **Kode Lengkap: Pelatihan, Evaluasi, Analisis, Plotting & Ekspor Model Deteksi TB**

# %%
import os
import random
import shutil
import numpy as np
import pandas as pd
from PIL import Image
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, DepthwiseConv2D, Add, GlobalAveragePooling2D, Dense, Softmax
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import OrthogonalRegularizer
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

# Pastikan TensorFlow versi yang sesuai (opsional)
# !pip install tensorflow==2.15.0
# Pastikan Pandas terinstal jika belum ada
# !pip install pandas

# %% [markdown]
# **1. Setup dan Persiapan Data**

# %%
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Definisikan path ke dataset Anda di Google Drive
normal_dir = "/content/drive/MyDrive/Kuliah/xray_tb_normal/Normal"
tb_dir = "/content/drive/MyDrive/Kuliah/xray_tb_normal/Tuberculosis"
output_viz_dir = "/content/drive/MyDrive/Kuliah/xray_tb_normal/visualizations" # Direktori untuk menyimpan visualisasi
base_save_dir = "/content/drive/MyDrive/Kuliah/xray_tb_normal/exported_models" # Direktori untuk menyimpan model yang diekspor
history_csv_path = "/content/drive/MyDrive/Kuliah/xray_tb_normal/training_history.csv" # Path untuk menyimpan riwayat pelatihan CSV


# Buat direktori output visualisasi jika belum ada
if not os.path.exists(output_viz_dir):
    os.makedirs(output_viz_dir)
    print(f"Direktori visualisasi '{output_viz_dir}' dibuat.")

# Buat direktori base untuk model yang diekspor jika belum ada
if not os.path.exists(base_save_dir):
    os.makedirs(base_save_dir)
    print(f"Direktori base model '{base_save_dir}' dibuat.")


# Dapatkan daftar nama file gambar
normal_files = [os.path.join(normal_dir, f) for f in os.listdir(normal_dir) if f.endswith('.png') or f.endswith('.jpg')]
tb_files = [os.path.join(tb_dir, f) for f in os.listdir(tb_dir) if f.endswith('.png') or f.endswith('.jpg')]

# Buat label
normal_labels = [0] * len(normal_files)
tb_labels = [1] * len(tb_files)

# Gabungkan file dan label
all_files = normal_files + tb_files
all_labels = normal_labels + tb_labels

# Acak dataset yang digabungkan
combined_data = list(zip(all_files, all_labels))
random.shuffle(combined_data)
all_files, all_labels = zip(*combined_data)

# Tentukan jumlah sampel untuk pelatihan dan pengujian sesuai jurnal
num_train_per_class = 500
num_test_per_class = 200

# Pisahkan kembali file berdasarkan kelas untuk pemisahan yang terkontrol
normal_files_shuffled = [f for f, label in zip(all_files, all_labels) if label == 0]
tb_files_shuffled = [f for f, label in zip(all_files, all_labels) if label == 1]

# Ambil jumlah sampel yang ditentukan untuk train dan test
train_normal_files = normal_files_shuffled[:num_train_per_class]
test_normal_files = normal_files_shuffled[num_train_per_class:num_train_per_class + num_test_per_class]

train_tb_files = tb_files_shuffled[:num_train_per_class]
test_tb_files = tb_files_shuffled[num_train_per_class:num_train_per_class + num_test_per_class]

# Gabungkan file train dan test dan buat label yang sesuai
train_files = train_normal_files + train_tb_files
test_files = test_normal_files + test_tb_files

train_labels = [0] * len(train_normal_files) + [1] * len(train_tb_files)
test_labels = [0] * len(test_normal_files) + [1] * len(test_tb_files)

print(f"Jumlah sampel pelatihan: {len(train_files)}")
print(f"Jumlah sampel pengujian: {len(test_files)}")

# Definisikan dimensi gambar
img_width, img_height = 224, 224

def load_and_preprocess_image(filepath, label, target_size=(img_width, img_height)):
    """Memuat, mengubah ukuran, dan memproses gambar tunggal."""
    try:
        img = load_img(filepath, target_size=target_size)
        img_array = img_to_array(img)
        # Normalisasi nilai piksel ke rentang [0, 1]
        img_array = img_array / 255.0
        return img_array, label
    except Exception as e:
        print(f"Error memuat atau memproses gambar {filepath}: {e}")
        return None, None

# Memuat dan memproses data pelatihan
print("Memuat dan memproses gambar pelatihan...")
train_images = []
train_labels_processed = []
for filepath, label in zip(train_files, train_labels):
    img_array, processed_label = load_and_preprocess_image(filepath, label)
    if img_array is not None:
        train_images.append(img_array)
        train_labels_processed.append(processed_label)

train_images = np.array(train_images)
train_labels_processed = to_categorical(train_labels_processed, num_classes=2)

# Memuat dan memproses data pengujian
print("Memuat dan memproses gambar pengujian...")
test_images = []
test_labels_processed = []
for filepath, label in zip(test_files, test_labels):
     img_array, processed_label = load_and_preprocess_image(filepath, label)
     if img_array is not None:
        test_images.append(img_array)
        test_labels_processed.append(processed_label)

test_images = np.array(test_images)
test_labels_processed = to_categorical(test_labels_processed, num_classes=2)

print(f"Bentuk gambar pelatihan: {train_images.shape}")
print(f"Bentuk label pelatihan: {train_labels_processed.shape}")
print(f"Bentuk gambar pengujian: {test_images.shape}")
print(f"Bentuk label pengujian: {test_labels_processed.shape}")


# %% [markdown]
# **2. Definisi Model MobileNetV2-based**

# %%
def conv_block(inputs, filters, kernel_size, strides):
    x = Conv2D(filters, kernel_size, strides=strides, padding='same', use_bias=False)(inputs)
    x = BatchNormalization()(x)
    x = ReLU(6.)(x)
    return x

def inverted_residual_block(inputs, expansion, filters, strides, alpha, block_id):
    in_channels = inputs.shape[-1]
    hidden_channels = int(in_channels * expansion)
    pointwise_conv_filters = int(filters * alpha)

    x = inputs
    residual = inputs

    # Expand
    if expansion != 1:
        x = conv_block(x, hidden_channels, kernel_size=1, strides=1)

    # Depthwise Conv
    x = DepthwiseConv2D(kernel_size=3, strides=strides, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU(6.)(x)

    # Project
    x = Conv2D(pointwise_conv_filters, kernel_size=1, strides=1, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)

    # Add residual if strides is 1 and input/output channels are the same
    if strides == 1 and in_channels == pointwise_conv_filters:
        x = Add()([residual, x])

    return x

def build_mobilenetv2_based_model(input_shape=(224, 224, 3), num_classes=2, alpha=1.0, resolution_multiplier=1.0):
    inputs = Input(shape=input_shape)
    x = conv_block(inputs, filters=int(32 * alpha), kernel_size=3, strides=(2, 2))
    x = inverted_residual_block(x, expansion=1, filters=16, strides=1, alpha=alpha, block_id=0)
    x = inverted_residual_block(x, expansion=6, filters=24, strides=2, alpha=alpha, block_id=1)
    x = inverted_residual_block(x, expansion=6, filters=24, strides=1, alpha=alpha, block_id=2)
    x = inverted_residual_block(x, expansion=6, filters=32, strides=2, alpha=alpha, block_id=3)
    x = inverted_residual_block(x, expansion=6, filters=32, strides=1, alpha=alpha, block_id=4)
    x = inverted_residual_block(x, expansion=6, filters=32, strides=1, alpha=alpha, block_id=5)
    x = inverted_residual_block(x, expansion=6, filters=64, strides=2, alpha=alpha, block_id=6)
    x = inverted_residual_block(x, expansion=6, filters=64, strides=1, alpha=alpha, block_id=7)
    x = inverted_residual_block(x, expansion=6, filters=64, strides=1, alpha=alpha, block_id=8)
    x = inverted_residual_block(x, expansion=6, filters=64, strides=1, alpha=alpha, block_id=9)
    x = inverted_residual_block(x, expansion=6, filters=96, strides=1, alpha=alpha, block_id=10)
    x = inverted_residual_block(x, expansion=6, filters=96, strides=1, alpha=alpha, block_id=11)
    x = inverted_residual_block(x, expansion=6, filters=96, strides=1, alpha=alpha, block_id=12)
    x = inverted_residual_block(x, expansion=6, filters=160, strides=2, alpha=alpha, block_id=13)
    x = inverted_residual_block(x, expansion=6, filters=160, strides=1, alpha=alpha, block_id=14)
    x = inverted_residual_block(x, expansion=6, filters=160, strides=1, alpha=alpha, block_id=15)
    x = inverted_residual_block(x, expansion=6, filters=320, strides=1, alpha=alpha, block_id=16)
    x = conv_block(x, filters=int(1280 * alpha), kernel_size=1, strides=1)
    x = GlobalAveragePooling2D()(x)
    outputs = Dense(num_classes, activation='softmax')(x) # OSL konseptual

    if resolution_multiplier != 1.0:
      pass

    model = Model(inputs=inputs, outputs=outputs)
    return model

model = build_mobilenetv2_based_model()

# %% [markdown]
# **3. Kompilasi Model**

# %%
initial_learning_rate = 0.0001
model.compile(optimizer=Adam(learning_rate=initial_learning_rate),
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.summary()

# %% [markdown]
# **4. Pelatihan Model**

# %%
epochs = 50
batch_size = 8

print("Memulai pelatihan model...")
history = model.fit(train_images, train_labels_processed,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_data=(test_images, test_labels_processed))
print("Pelatihan model selesai.")

# %% [markdown]
# **8. Menyimpan Riwayat Pelatihan ke CSV**

# %%
# Ubah objek history menjadi dictionary
history_dict = history.history

# Buat Pandas DataFrame dari dictionary history
df_history = pd.DataFrame(history_dict)

# Menyimpan DataFrame ke file CSV dengan delimiter koma
try:
    df_history.to_csv(history_csv_path, index=False, sep=',') # index=False agar tidak menyimpan index DataFrame
    print(f"Riwayat pelatihan berhasil disimpan ke: {history_csv_path}")
except Exception as e:
    print(f"Gagal menyimpan riwayat pelatihan ke CSV: {e}")


# %% [markdown]
# **5. Evaluasi Model Dasar**

# %%
print("Mengevaluasi model pada data pengujian...")
loss, accuracy = model.evaluate(test_images, test_labels_processed, verbose=0)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

# %% [markdown]
# **6. Analisis Performa Detail dan Plotting Visualisasi**

# %%
# Dapatkan prediksi pada data pengujian
y_pred_probs = model.predict(test_images)
y_pred_classes = np.argmax(y_pred_probs, axis=1)
y_true_classes = np.argmax(test_labels_processed, axis=1)

# Hitung dan Plot Confusion Matrix
cm = confusion_matrix(y_true_classes, y_pred_classes)
print("\nConfusion Matrix:")
print(cm)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Normal', 'Tuberculosis'], yticklabels=['Normal', 'Tuberculosis'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
# Simpan plot confusion matrix
plt.savefig(os.path.join(output_viz_dir, 'confusion_matrix.png'))
plt.show()

# Hitung Classification Report
print("\nClassification Report:")
print(classification_report(y_true_classes, y_pred_classes, target_names=['Normal', 'Tuberculosis']))

# Hitung kurva ROC dan AUC, dan Plot ROC Curve
fpr, tpr, thresholds = roc_curve(y_true_classes, y_pred_probs[:, 1])
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
# Simpan plot ROC curve
plt.savefig(os.path.join(output_viz_dir, 'roc_curve.png'))
plt.show()

# Plot Training and Validation Loss
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
# Simpan plot loss
plt.savefig(os.path.join(output_viz_dir, 'loss_plot.png'))
plt.show()

# Plot Training and Validation Accuracy
plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
# Simpan plot accuracy
plt.savefig(os.path.join(output_viz_dir, 'accuracy_plot.png'))
plt.show()


# %% [markdown]
# **7. Menyimpan Model dalam Berbagai Format (SavedModel, H5, TFLite)**

# %%
# Tentukan path spesifik untuk setiap format
savedmodel_path = os.path.join(base_save_dir, "saved_model")
h5_path = os.path.join(base_save_dir, "model.h5")
tflite_path = os.path.join(base_save_dir, "model.tflite") # Untuk Android

# 7.1. Menyimpan model dalam format TensorFlow SavedModel (Direkomendasikan)
print(f"\nMenyimpan model dalam format SavedModel ke: {savedmodel_path}")
try:
    model.save(savedmodel_path)
    print("Model berhasil disimpan dalam format SavedModel.")
except Exception as e:
    print(f"Gagal menyimpan model dalam format SavedModel: {e}")

# 7.2. Menyimpan model dalam format H5 (Legacy)
print(f"\nMenyimpan model dalam format H5 ke: {h5_path}")
try:
    model.save(h5_path)
    print("Model berhasil disimpan dalam format H5.")
except Exception as e:
    print(f"Gagal menyimpan model dalam format H5: {e}")

# 7.3. Mengkonversi dan Menyimpan model dalam format TensorFlow Lite (untuk Android)
print(f"\nMengkonversi dan menyimpan model dalam format TensorFlow Lite ke: {tflite_path}")
try:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    print("Model berhasil dikonversi dan disimpan dalam format TensorFlow Lite.")
except Exception as e:
    print(f"Gagal mengkonversi atau menyimpan model TensorFlow Lite: {e}")

print("\nProses ekspor model dan penyimpanan riwayat pelatihan selesai.")
