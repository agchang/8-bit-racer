from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras.optimizers import RMSprop


def build_model(input_shape, num_actions):
    """Builds a DQN as per the DeepMind architecture, which is a
    convolutional neural net that takes in input with input_shape,
    which should be (# images, image_height, image_width), and
    num_action output nodes, which is the current value of Q*(s,a).
    """
    model = Sequential()
    # 32 8x8 filters with stride 4
    model.add(
        Conv2D(
            32,
            (8, 8),
            strides=(4, 4),
            padding="same",
            activation="relu",
            input_shape=input_shape,
        )
    )

    # 64 4x4 filters with stride 2
    model.add(Conv2D(64, (4, 4), strides=(2, 2), padding="same", activation="relu"))

    # 64 3x3 filters with stride 1
    model.add(Conv2D(64, (3, 3), strides=(1, 1), padding="same", activation="relu"))

    model.add(Flatten())

    # Fully connected 512 neurons
    model.add(Dense(512, activation="relu"))

    # Output nodes, one per action
    model.add(Dense(num_actions))

    rmsprop = RMSprop(lr=1e-6)
    model.compile(loss="mse", optimizer=rmsprop)
    return model
