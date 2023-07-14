import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image

import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.utils import to_categorical


def ModelTraining():
    # Preprcessing
    (X_train, y_train), (X_val, y_val) = cifar10.load_data()


    # Filter the dataset to include only the selected categories
    selected_categories = [0, 1, 8, 9]  # Airplane, Automobile, Ship, Truck
    train_indices = np.isin(y_train, selected_categories).flatten()
    val_indices = np.isin(y_val, selected_categories).flatten()
    X_train = X_train[train_indices] / 255
    X_val = X_val[val_indices] / 255

    # Reassign labels to consecutive integers
    y_train = y_train[train_indices]
    y_val = y_val[val_indices]
    label_mapping = {category: i for i, category in enumerate(selected_categories)}
    y_train = np.vectorize(label_mapping.get)(y_train)
    y_val = np.vectorize(label_mapping.get)(y_val)

    # Convert labels to categorical format
    num_classes = len(selected_categories)
    y_train = to_categorical(y_train, num_classes)
    y_val = to_categorical(y_val, num_classes)



    model = Sequential([
        Flatten(input_shape=(32,32,3)),
        Dense(1000, activation='relu'),
        Dense(4, activation='softmax')
    ])

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, batch_size=64, epochs=10, validation_data=(X_val, y_val))
    model.save('vehicle_model.h5')


def main():
    st.title('Vehicle Image Classifer')
    st.write('Upload any image that you think fits into one of the classes (Airplane, Automobile, Ship, Truck)')

    file = st.file_uploader('Upload Image', type=['jpg'])
    if file:
        image = Image.open(file)
        st.image(image, use_column_width = True)

        resized_image = image.resize((32, 32))
        img_array = np.array(resized_image) / 255
        img_array = img_array.reshape((1, 32, 32, 3))

        
        model = tf.keras.models.load_model('vehicle_model.h5')

        predictions = model.predict(img_array)
        vehicle_classes = ['airplane', 'automobile', 'ship', 'truck']

        fig, ax = plt.subplots()
        y_pos = np.arange(len(vehicle_classes))
        ax.barh(y_pos, predictions[0], align='center')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(vehicle_classes)
        ax.invert_yaxis()
        ax.set_xlabel('Probability')
        ax.set_title('Predictions')
        
        st.pyplot(fig)

    else:
        st.text('No File Uploaded Yet')


if __name__ == '__main__':
    print("NumPy version:", np.__version__)
    print("Matplotlib version:", plt.__version__)
    print("Streamlit version:", st.__version__)
    print("Pillow version:", Image.__version__)
    print("TensorFlow version:", tf.__version__)
    main()