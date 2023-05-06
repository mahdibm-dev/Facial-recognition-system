from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout

def build_model(input_shape, num_classes):
    # Define the input layer
    input_layer = Input(shape=input_shape)

    # Add convolutional and pooling layers
    x = Conv2D(32, (3, 3), padding='same', activation='relu')(input_layer)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Flatten the output from the convolutional layers
    x = Flatten()(x)

    # Add fully connected layers with dropout
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)

    # Add the output layer with softmax activation for the classes
    output_layer = Dense(num_classes, activation='softmax')(x)

    # Define the model inputs and outputs
    model = Model(inputs=input_layer, outputs=output_layer)

    return model
