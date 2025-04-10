# Step 1: Import necessary libraries
import flwr as fl
import tensorflow as tf
import matplotlib.pyplot as plt

# Step 2: Loading dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Step 3: Normalization
x_train, x_test = x_train / 255.0, x_test / 255.0

# Step 4: Define the Model
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Step 5: Compile model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Step 6: Define the Client
class SimpleClient(fl.client.NumPyClient):
    def __init__(self):
        self.accuracy_list = []  # Store accuracy values after each round

    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        model.fit(x_train, y_train, epochs=1, batch_size=32)
        return model.get_weights(), len(x_train), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(x_test, y_test)
        self.accuracy_list.append(accuracy)  # Append accuracy to the list
        return loss, len(x_test), {"accuracy": accuracy}

# Step 7: Start the client and plot accuracy
client = SimpleClient()

def client_training():
    fl.client.start_numpy_client(
        server_address="127.0.0.1:8080",
        client=client,
    )

# Run client training
client_training()

# Step 8: Plot accuracy vs. number of rounds
rounds = range(1, len(client.accuracy_list) + 1)
plt.plot(rounds, client.accuracy_list, marker='o')
plt.title('Accuracy vs. Number of Rounds for Clients')
plt.xlabel('Rounds')
plt.ylabel('Accuracy')
plt.grid(True)
plt.show()
