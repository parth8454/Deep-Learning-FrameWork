import sys
import time
from sklearn.datasets import load_digits
from nn.model import MLP
from nn.optim import Adam
from nn.losses import MSELoss

# herre we manually inc the limit of python recursion otherwise it will just throw an err...
#  you can comment the following line and run the code to se the error yourself
sys.setrecursionlimit(20000)

def train_mnist():
    print("Loading Digits dataset...")
    digits = load_digits()
    
    X_train_raw = digits.data[:80]
    y_train_raw = digits.target[:80]
    
    X_val_raw = digits.data[80:100]
    y_val_raw = digits.target[80:100]

    X_train = [[float(p) / 16.0 for p in x] for x in X_train_raw]
    X_val = [[float(p) / 16.0 for p in x] for x in X_val_raw]

    def one_hot_encode(labels):
        encoded = []
        for label in labels:
            one_hot = [0.0] * 10
            one_hot[label] = 1.0
            encoded.append(one_hot)
        return encoded

    y_train = one_hot_encode(y_train_raw)
    y_val = one_hot_encode(y_val_raw)

    nn = MLP(64, [16, 10])
    optimizer = Adam(nn.parameters(), lr=0.01)
    criterion = MSELoss()

    print(f"Network has {len(nn.parameters())} total trainable parameters.")

    epochs = 20

    for k in range(epochs):
        ypred_train = [nn(x) for x in X_train]
        
        ypred_train_flat = [val for sublist in ypred_train for val in sublist]
        y_train_flat = [val for sublist in y_train for val in sublist]
        
        train_loss = criterion(y_train_flat, ypred_train_flat)
        
        train_correct = 0
        for i in range(len(X_train)):
            pred_label = max(range(10), key=lambda j: ypred_train[i][j].data)
            if pred_label == y_train_raw[i]:
                train_correct += 1
        train_acc = (train_correct / len(X_train)) * 100.0

        optimizer.zero_grad() 
        train_loss.backward()
        optimizer.step()

        ypred_val = [nn(x) for x in X_val]
        
        ypred_val_flat = [val for sublist in ypred_val for val in sublist]
        y_val_flat = [val for sublist in y_val for val in sublist]
        
        val_loss = criterion(y_val_flat, ypred_val_flat)
        
        val_correct = 0
        for i in range(len(X_val)):
            pred_label = max(range(10), key=lambda j: ypred_val[i][j].data)
            if pred_label == y_val_raw[i]:
                val_correct += 1
        val_acc = (val_correct / len(X_val)) * 100.0
            
        print(f"Epoch {k:2d} | "
              f"Train Loss: {train_loss.data:.4f} | Train Acc: {train_acc:5.1f}% | "
              f"Val Loss: {val_loss.data:.4f} | Val Acc: {val_acc:5.1f}%")

if __name__ == "__main__":
    train_mnist()