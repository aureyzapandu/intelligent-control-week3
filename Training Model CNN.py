from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

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
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

seg_train = train_datagen.flow_from_directory(
    'seg_train/seg_train',  # Ganti dengan path ke folder data training
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)

seg_test = test_datagen.flow_from_directory(
    'seg_test/seg_test',  # Ganti dengan path ke folder data testing
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)

# Training model
model.fit(seg_train, validation_data=seg_test, epochs=5)

# Simpan model
model.save('cnn_model.h5')

