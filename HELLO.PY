# A compact, from-scratch 2-layer neural network (≈100 lines) trained on sklearn's digits dataset.
# It uses only numpy + sklearn for data loading/splitting and reports accuracy.
# Copy-paste this into a file `two_layer_nn_digits.py` and run `python three_layer_nn_digits.py`.
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score

np.random.seed(42)

# --- Data ---
digits = load_digits()
X = digits.data / 16.0            # scale inputs (0..1)
y = digits.target.reshape(-1, 1)  # shape (n_samples,1)

enc = OneHotEncoder(sparse=False, categories='auto')
Y = enc.fit_transform(y)          # one-hot targets (n_samples, 10)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=42)

# --- Network hyperparams ---
n_inputs = X.shape[1]
n_hidden = 64
n_outputs = Y.shape[1]
lr = 0.5
epochs = 200
batch_size = 64
reg = 1e-3  # L2 regularization

# --- Helpers ---
def softmax(z):
    z = z - np.max(z, axis=1, keepdims=True)
    exp = np.exp(z)
    return exp / np.sum(exp, axis=1, keepdims=True)

def relu(z):
    return np.maximum(0, z)

def drelu(z):
    return (z > 0).astype(float)

def cross_entropy_loss(pred, target):
    # pred: probabilities, target: one-hot
    n = pred.shape[0]
    clipped = np.clip(pred, 1e-12, 1 - 1e-12)
    return -np.sum(target * np.log(clipped)) / n

# --- Parameter initialization (He for ReLU) ---
W1 = np.random.randn(n_inputs, n_hidden) * np.sqrt(2. / n_inputs)
b1 = np.zeros((1, n_hidden))
W2 = np.random.randn(n_hidden, n_outputs) * np.sqrt(2. / n_hidden)
b2 = np.zeros((1, n_outputs))

# --- Training loop (mini-batch SGD) ---
n_samples = X_train.shape[0]
for epoch in range(1, epochs + 1):
    # shuffle
    perm = np.random.permutation(n_samples)
    Xs = X_train[perm]
    Ys = Y_train[perm]
    for i in range(0, n_samples, batch_size):
        xb = Xs[i:i+batch_size]
        yb = Ys[i:i+batch_size]
        # forward
        z1 = xb.dot(W1) + b1
        a1 = relu(z1)
        z2 = a1.dot(W2) + b2
        probs = softmax(z2)

        # loss (for monitoring / numeric stability)
        # backward
        n_b = xb.shape[0]
        dz2 = (probs - yb) / n_b
        dW2 = a1.T.dot(dz2) + reg * W2
        db2 = np.sum(dz2, axis=0, keepdims=True)

        da1 = dz2.dot(W2.T)
        dz1 = da1 * drelu(z1)
        dW1 = xb.T.dot(dz1) + reg * W1
        db1 = np.sum(dz1, axis=0, keepdims=True)

        # parameter update (SGD)
        W2 -= lr * dW2
        b2 -= lr * db2
        W1 -= lr * dW1
        b1 -= lr * db1

    # monitor
    if epoch % 20 == 0 or epoch == 1 or epoch == epochs:
        # compute train loss and accuracy
        a1_train = relu(X_train.dot(W1) + b1)
        train_probs = softmax(a1_train.dot(W2) + b2)
        train_loss = cross_entropy_loss(train_probs, Y_train) + 0.5 * reg * (np.sum(W1*W1)+np.sum(W2*W2))
        train_pred = np.argmax(train_probs, axis=1)
        train_true = np.argmax(Y_train, axis=1)
        train_acc = accuracy_score(train_true, train_pred)

        a1_test = relu(X_test.dot(W1) + b1)
        test_probs = softmax(a1_test.dot(W2) + b2)
        test_loss = cross_entropy_loss(test_probs, Y_test) + 0.5 * reg * (np.sum(W1*W1)+np.sum(W2*W2))
        test_pred = np.argmax(test_probs, axis=1)
        test_true = np.argmax(Y_test, axis=1)
        test_acc = accuracy_score(test_true, test_pred)

        print(f"Epoch {epoch:3d} | train loss {train_loss:.4f} acc {train_acc:.4f} | test loss {test_loss:.4f} acc {test_acc:.4f}")

# final evaluation
a1_test = relu(X_test.dot(W1) + b1)
test_probs = softmax(a1_test.dot(W2) + b2)
test_pred = np.argmax(test_probs, axis=1)
test_true = np.argmax(Y_test, axis=1)
print("\nFinal test accuracy:", accuracy_score(test_true, test_pred))

