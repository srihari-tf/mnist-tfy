
import mlfoundry as mlf

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import Callback

mlf_api = mlf.get_client()
run = mlf_api.create_run('mnist-keras')

# Model / data parameters
num_classes = 10
input_shape = (28, 28, 1)

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
# Make sure images have shape (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


class MetricsLogCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        run.log_metrics(logs, epoch)  


BATCH_SIZE = 128
EPOCHS = 5

run.log_params({'batch_size': BATCH_SIZE, 'epochs': EPOCHS})

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=0.1, callbacks=[MetricsLogCallback()])

run.log_model(model, 'keras')
run.end()
print(run.run_id)
