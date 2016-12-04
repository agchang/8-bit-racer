from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import RMSprop
from keras.layers.convolutional import Convolution2D

def buildModel(input_shape, num_actions):
    """Builds a DQN as per the DeepMind architecture, which is a 
    convolutional neural net that takes in input with input_shape, 
    which should be (# images, image_height, image_width), and 
    num_action output nodes, which is the current value of Q*(s,a).
    """
    model = Sequential()
    # 32 8x8 filters with stride 4
    model.add(Convolution2D(32, 8, 8, subsample=(4,4), \
	border_mode='same',input_shape=(NUM_IMAGES, HEIGHT, WIDTH)))
    # Followed by a RELU
    model.add(Activation('relu'))

    # 64 4x4 filters with stride 2
    model.add(Convolution2D(64, 4, 4, subsample=(2,2), \
	border_mode='same'))
    # Followed by a RELU
    model.add(Activation('relu'))

    # 64 3x3 filters with stride 1
    model.add(Convolution2D(64, 3, 3, subsample=(1,1), \
            border_mode='same'))

    # Followed by a RELU
    model.add(Activation('relu'))
    model.add(Flatten())
    # Fully connected 512 neurons
    model.add(Dense(512))
    # Followed by a RELU
    model.add(Activation('relu'))
    # Output nodes, one per action
    model.add(Dense(NUM_ACTIONS))
   
    rmsprop = RMSprop(lr=1e-6)
    model.compile(loss='mse', optimizer=rmsprop) 
    return model
