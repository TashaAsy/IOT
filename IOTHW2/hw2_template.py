import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras import layers, Model, Input
from PIL import Image

# Model Definitions
def build_model1():
    model = tf.keras.Sequential([

        # Initial Convolution Layer
        layers.Conv2D(32, (3, 3), strides=(2, 2), activation='relu', padding='same', input_shape=(32, 32, 3)),
        layers.BatchNormalization(),

        # Convolution Layers
        layers.Conv2D(64, (3, 3), strides=(2, 2), activation='relu', padding='same'),
        layers.BatchNormalization(),

        layers.Conv2D(128, (3, 3), strides=(2, 2), activation='relu', padding='same'),
        layers.BatchNormalization(),

        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),

        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),

        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),

        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),

        # Pooling and Flattening
        layers.MaxPooling2D((4, 4), strides=(4, 4)),
        layers.Flatten(),

        # Dense Layers
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(10, activation='softmax')
    ])
    return model

def build_model2():
    model = tf.keras.Sequential([

        # Initial Convolution Layer
        layers.Conv2D(32, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='same', input_shape=(32, 32, 3)),
        layers.BatchNormalization(),

        # Separable Convolution Layers
        layers.SeparableConv2D(64, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='same'),
        layers.BatchNormalization(),

        layers.SeparableConv2D(128, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='same'),
        layers.BatchNormalization(),

        layers.SeparableConv2D(128, kernel_size=(3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),

        layers.SeparableConv2D(128, kernel_size=(3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),

        layers.SeparableConv2D(128, kernel_size=(3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),

        layers.SeparableConv2D(128, kernel_size=(3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),

        # Pooling and Flattening
        layers.MaxPooling2D((4, 4), strides=(4, 4)),
        layers.Flatten(),

        # Dense Layers
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(10, activation='softmax')
    ])
    return model

def build_model3():
    inputs = Input(shape=(32, 32, 3))
    
    # First Convolutional Block
    residual = layers.Conv2D(32, (3, 3), strides=(2, 2), activation='relu', padding='same')(inputs)
    conv1 = layers.BatchNormalization()(residual)
    conv1 = layers.Dropout(0.5)(conv1)

    # Second Convolutional Block
    conv2 = layers.Conv2D(64, (3, 3), strides=(2, 2), activation='relu', padding='same')(conv1)
    conv2 = layers.BatchNormalization()(conv2)
    conv2 = layers.Dropout(0.5)(conv2)

    # Third Convolutional Block
    conv3 = layers.Conv2D(128, (3, 3), strides=(2, 2), activation='relu', padding='same')(conv2)
    conv3 = layers.BatchNormalization()(conv3)
    conv3 = layers.Dropout(0.5)(conv3)

    # Residual Block 1
    shortcut1 = layers.Conv2D(128, (1, 1), strides=(4, 4))(residual)
    shortcut1 = layers.Add()([shortcut1, conv3])

    # Fourth Convolutional Block
    conv4 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(shortcut1)
    conv4 = layers.BatchNormalization()(conv4)
    conv4 = layers.Dropout(0.5)(conv4)

    # Fifth Convolutional Block
    conv5 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv4)
    conv5 = layers.BatchNormalization()(conv5)
    conv5 = layers.Dropout(0.5)(conv5)

    # Residual Block 2
    shortcut2 = layers.Add()([shortcut1, conv5])

    # Sixth Convolutional Block
    conv6 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(shortcut2)
    conv6 = layers.BatchNormalization()(conv6)
    conv6 = layers.Dropout(0.5)(conv6)

    # Seventh Convolutional Block
    conv7 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv6)
    conv7 = layers.BatchNormalization()(conv7)
    conv7 = layers.Dropout(0.5)(conv7)

    # Final Residual Addition
    shortcut3 = layers.Add()([shortcut2, conv7])

    # Pooling and Flattening
    pooling = layers.MaxPooling2D((4, 4), strides=(4, 4))(shortcut3)
    flatten = layers.Flatten()(pooling)

    dense = layers.Dense(128, activation='relu')(flatten)
    dense = layers.BatchNormalization()(dense)

    outputs = layers.Dense(10, activation='softmax')(dense)

    model = Model(inputs, outputs)
    return model

def build_model50k():
    model = tf.keras.Sequential([

        # Initial Convolution Layer
        layers.Conv2D(16, (5, 5), padding='same', activation='relu', input_shape=(32, 32, 3)),

        # Convolution Layers
        layers.Conv2D(8, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Flatten and Dense Layers
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])
    return model

if __name__ == '__main__':
    ########################################
    ## Add code here to Load the CIFAR10 data set
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    # Split training set into training and validation sets
    val_fraction = 0.1
    num_val_samples = int(len(train_images) * val_fraction)
    
    val_indices = np.random.choice(np.arange(len(train_images)), size=num_val_samples, replace=False)
    train_indices = np.setdiff1d(np.arange(len(train_images)), val_indices)
    val_images = train_images[val_indices, :, :, :]
    train_images = train_images[train_indices, :, :, :]

    val_labels = train_labels[val_indices]
    train_labels = train_labels[train_indices]

    train_labels = train_labels.squeeze()
    test_labels = test_labels.squeeze()
    val_labels = val_labels.squeeze()

    # Normalize the images
    train_images = train_images / 255.0
    test_images = test_images / 255.0
    val_images = val_images / 255.0

    ########################################
    
    # Build and train model 1
    model1 = build_model1()
    model1.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model1.summary()
    
    history1 = model1.fit(train_images, train_labels, epochs=50, validation_data=(val_images, val_labels))
    
    train_loss1, train_acc1 = model1.evaluate(train_images, train_labels)
    print('Training accuracy:', train_acc1)
    val_loss1, val_acc1 = model1.evaluate(val_images, val_labels)
    print('Validation accuracy:', val_acc1)
    test_loss1, test_acc1 = model1.evaluate(test_images, test_labels)
    print('Test accuracy:', test_acc1)

    model1.save('model1.h5')
    print("Model 1 saved")

    ########################################
    
    # Build, compile, and train model 2 (Depthwise Separable Convolutions)
    model2 = build_model2()
    model2.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model2.summary()
    
    history2 = model2.fit(train_images, train_labels, epochs=50, validation_data=(val_images, val_labels))
    
    train_loss2, train_acc2 = model2.evaluate(train_images, train_labels)
    print('Training accuracy:', train_acc2)
    val_loss2, val_acc2 = model2.evaluate(val_images, val_labels)
    print('Validation accuracy:', val_acc2)
    test_loss2, test_acc2 = model2.evaluate(test_images, test_labels)
    print('Test accuracy:', test_acc2)

    model2.save('model2.h5')
    print("Model 2 saved")

    ########################################
    
    # Build, compile, and train model 3 (Residual Connections)
    model3 = build_model3()
    model3.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model3.summary()

    history3 = model3.fit(train_images, train_labels, validation_data=(val_images, val_labels), epochs=50)
    train_loss3, train_acc3 = model3.evaluate(test_images, test_labels, verbose=2)
    print('Training accuracy:', train_acc3)
    val_loss3, val_acc3 = model3.evaluate(val_images, val_labels, verbose=2)
    print('Validation accuracy:', val_acc3)
    test_loss3, test_acc3 = model3.evaluate(test_images, test_labels, verbose=2)
    print('Test accuracy:', test_acc3)
    
    model3.save('model3.h5')
    print("Model 3 saved")

    ########################################

    # Build, compile, and train best model under 50k parameters
    best_model = build_model50k()
    best_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    best_model.summary()

    history_best = best_model.fit(train_images, train_labels, validation_data=(val_images, val_labels), epochs=50)
    train_loss_best, train_acc_best = best_model.evaluate(test_images, test_labels, verbose=2)
    print('Training accuracy:', train_acc_best)
    val_loss_best, val_acc_best = best_model.evaluate(val_images, val_labels, verbose=2)
    print('Validation accuracy:', val_acc_best)
    test_loss_best, test_acc_best = best_model.evaluate(test_images, test_labels, verbose=2)
    print('Test accuracy:', test_acc_best)

    best_model.save('best_model.h5')
    print("Best Model saved")
