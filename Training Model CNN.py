import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

def show_graphic():
    if not os.path.exists('training_history.png'):
        print("File grafik belum ada. Silakan lakukan training terlebih dahulu.")
        return
    img = plt.imread('training_history.png')
    plt.imshow(img)
    plt.axis('off')
    plt.title('Training History')
    plt.show()

def train_and_plot():
    # Definisi model CNN
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 3)),
        MaxPooling2D(2,2),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(6, activation='softmax')
    ])

    # Kompilasi model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Persiapan data pelatihan dan pengujian
    train_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)

    seg_train = train_datagen.flow_from_directory(
        r'G:\[Data Aureyza\Kuliah Au\SMT 7\8. Prak. Kontrol Cerdas\Week 3\intelligent-control-week3-main\seg_train\seg_train',
        target_size=(150, 150),
        batch_size=32,
        class_mode='categorical'
    )

    seg_test = test_datagen.flow_from_directory(
        r'G:\[Data Aureyza\Kuliah Au\SMT 7\8. Prak. Kontrol Cerdas\Week 3\intelligent-control-week3-main\seg_test\seg_test',
        target_size=(150, 150),
        batch_size=32,
        class_mode='categorical'
    )

    # Training model
    history = model.fit(seg_train, validation_data=seg_test, epochs=10)

    # Simpan model
    model.save('cnn_model.h5')

    # Plot grafik akurasi dan loss
    plt.figure(figsize=(12,5))

    # Plot akurasi
    plt.subplot(1,2,1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Accuracy per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot loss
    plt.subplot(1,2,2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

if __name__ == "__main__":
    print("Pilih opsi:")
    print("1. Training dan tampilkan grafik")
    print("2. Hanya tampilkan grafik")
    opsi = input("Masukkan pilihan (1/2): ").strip()
    if opsi == "1":
        train_and_plot()
    elif opsi == "2":
        show_graphic()
    else:
        print("Pilihan tidak valid.")

