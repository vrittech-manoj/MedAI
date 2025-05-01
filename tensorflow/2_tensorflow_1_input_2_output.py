import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

# Single input
input_layer = Input(shape=(16,))  # for example, 16 features

# Shared base
x = Dense(64, activation='relu')(input_layer)
x = Dense(32, activation='relu')(x)

# Output 1 (e.g., binary classification)
output1 = Dense(1, activation='sigmoid', name='output_class')(x)

# Output 2 (e.g., regression)
output2 = Dense(1, activation='linear', name='output_regression')(x)

# Create model
model = Model(inputs=input_layer, outputs=[output1, output2])

# Compile with separate losses
model.compile(
    optimizer='adam',
    loss={
        'output_class': 'binary_crossentropy',
        'output_regression': 'mse'
    },
    metrics={
        'output_class': 'accuracy',
        'output_regression': 'mae'
    }
)

# Show model summary
model.summary()
