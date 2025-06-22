import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

# 1️⃣ Input layer
input_layer = Input(shape=(16,))  # For example, 16 features

# 2️⃣ Shared hidden layers
x = Dense(64, activation='relu')(input_layer)
x = Dense(32, activation='relu')(x)

# 3️⃣ Output 1 - Classification (binary)
output1 = Dense(1, activation='sigmoid', name='classification_output')(x)

# 4️⃣ Output 2 - Regression
output2 = Dense(1, activation='linear', name='regression_output')(x)

# 5️⃣ Define model
model = Model(inputs=input_layer, outputs=[output1, output2])

# 6️⃣ Compile with separate loss functions
model.compile(
    optimizer='adam',
    loss={
        'classification_output': 'binary_crossentropy',
        'regression_output': 'mse'
    },
    metrics={
        'classification_output': 'accuracy',
        'regression_output': 'mae'
    }
)

# 7️⃣ Summary
model.summary()
