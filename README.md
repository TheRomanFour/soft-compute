# soft-compute


MREŽA RADIJALNIH BAZNIH
FUNKCIJA 2/2
• Sastoje se od tri sloja (ulaza, skrivenog sloja sa
RBF te izlaznog sloja)
• Svaki neuron skrivenog sloja predstavlja prototip
(centroid) ulaznih podataka
▪ Izračunava se udaljenost između ulaza i
pojedinog prototipa te se izračun koristi kao
parametar radijalne bazne funkcije (npr.
Gaussove funkcije)



```
import tensorflow as tf

sampling_rate = 6 # Uzmimo svaki 6. point (sat vremena)
sequence_length = 120 # 5 dana
delay = sampling_rate * (sequence_length + 24 - 1)
batch_size = 256

# Train
train_dataset = tf.keras.utils.timeseries_dataset_from_array(
    raw_data[:-delay],
    targets = temperature[delay:],
    sampling_rate = sampling_rate,
    sequence_length = sequence_length,
    shuffle = True,
    batch_size = batch_size,
    start_index = 0,
    end_index = num_train_samples
)

# Val
val_dataset = tf.keras.utils.timeseries_dataset_from_array( #
    raw_data[:-delay],
    targets = temperature[delay:],
    sampling_rate = sampling_rate,
    sequence_length = sequence_length,
    shuffle = True,
    batch_size = batch_size,
    start_index = num_train_samples, #
    end_index = num_train_samples + num_val_samples #
)

# Test
test_dataset = tf.keras.utils.timeseries_dataset_from_array( #
    raw_data[:-delay],
    targets = temperature[delay:],
    sampling_rate = sampling_rate,
    sequence_length = sequence_length,
    shuffle = True,
    batch_size = batch_size,
    start_index = num_train_samples + num_val_samples, #
 #
)

```

```
# Izvođenje 11 minuta
# Dense Model
inputs = tf.keras.Input(
    shape = (sequence_length, raw_data.shape[-1])
)

x = tf.keras.layers.Flatten()(inputs)

x = tf.keras.layers.Dense(16, activation="relu")(x)

outputs = tf.keras.layers.Dense(1)(x)

model = tf.keras.Model(inputs, outputs)

# Kako bismo spremili model s najboljom performansom
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        "jena_dense.keras",
        save_best_only = True
    )
]

# Kompajliranje modela
model.compile(
    optimizer = "rmsprop",
    loss = "mse",
    metrics = ["mae"]
)

# Trening modela
history = model.fit(
    train_dataset,
    epochs = 10,
    validation_data = val_dataset,
    callbacks = callbacks
)

# Učitavanje najboljeg modela s obzirom na MAE metriku
model = tf.keras.models.load_model("jena_dense.keras")

# Evaluacija
print("Test MAE MLP modela: {}".format(model.evaluate(test_dataset)[1]))
```

