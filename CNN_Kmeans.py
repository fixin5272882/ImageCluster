import torch
from torch import nn
import numpy as np
import pandas
import matplotlib.pyplot as plt
import seaborn as sns
from torch.nn import functional as F
from keras.datasets import mnist

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

(X_train, y_train), (X_test, y_test) = mnist.load_data()
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

X_train = torch.from_numpy((X_train.reshape(-1, 1, 28, 28)).astype(np.float32))
X_test = torch.from_numpy((X_test.reshape(-1, 1, 28, 28)).astype(np.float32))
y_train = torch.from_numpy(y_train)
y_test = torch.from_numpy(y_test)
X_cluster = X_train[:10000]
y_cluster = y_train[:10000]
X_train = X_train[10000:]
y_train = y_train[10000:]
X_train.shape, X_test.shape, X_cluster.shape, y_train.shape, y_test.shape, y_cluster.shape

# hyperparameters
learning_rate = 0.001
batch_size = 64
n_epochs = 30
in_channels = 1
n_outputs = 10

train_ds = torch.utils.data.TensorDataset(X_train, y_train)
train_loader = torch.utils.data.DataLoader(dataset = train_ds, batch_size = batch_size, shuffle = True)
test_ds = torch.utils.data.TensorDataset(X_test, y_test)
test_loader = torch.utils.data.DataLoader(dataset = test_ds, batch_size = batch_size, shuffle = True)

class ConvolutionalNeuralNet(nn.Module):
    def __init__(self, in_channels = 1, n_outputs = 10):
        super(ConvolutionalNeuralNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = in_channels, out_channels = 8, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1)) # 28 -> 14
        self.conv2 = nn.Conv2d(in_channels = 8, out_channels = 16, kernel_size = (5, 5), stride = (1, 1), padding = (1, 1)) # 14 -> 7
        self.fc1 = nn.Linear(16 * 6 * 6, 10)
        self.pool = nn.MaxPool2d(kernel_size = (2, 2), stride = (2, 2))
        
    def forward(self, x, extract_feature = False):
        out = F.relu(self.conv1(x))
        out = self.pool(out)
        out = F.relu(self.conv2(out))
        out = self.pool(out)
        
        if extract_feature:
            return out.reshape(out.shape[0], -1)
        
        out = out.reshape(out.shape[0], -1)
        out = self.fc1(out)
        return out
    
# model - Convolutional
model = ConvolutionalNeuralNet().to(device)

# loss
critirion = nn.CrossEntropyLoss()

# optimizer
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

def predict(model, X_batch):
    X_batch = X_batch.to(device)
    y_pred = model(X_batch)
    return y_pred.max(dim = 1)[1]

def batch_accuracy(model, X_batch, y_batch):
    X_batch = X_batch.to(device)
    y_batch = y_batch.to(device)
    y_pred = predict(model, X_batch)
    return (y_pred == y_batch).sum() / batch_size

def val_accuracy(model, test_loader):
    sum = 0
    mx = 0
    for i, (samples, labels) in enumerate(test_loader):
        mx = max(mx, i)
        sum += batch_accuracy(model, samples, labels)
    return sum / (mx + 1)

for epoch in range(10):
    for i, (samples, labels) in enumerate(train_loader):
        
        samples = samples.to(device)
        labels = labels.to(device)
        # forward
        predictions = model(samples)
        loss = critirion(predictions, labels)
        
        # backward
        loss.backward()
        
        # optimizer
        optimizer.step()
        optimizer.zero_grad()
    if (epoch + 1) % 10 == 0:
        print(f"epoch = {epoch + 1} | loss = {loss:.5f} | train accuracy = {val_accuracy(model, train_loader):.4f}  | val accuracy = {val_accuracy(model, test_loader):.4f}")

    model.eval()
y_pred = predict(model, X_test).to('cpu')
p = np.random.randint(1, 10000)
plt.imshow(X_test[p].reshape(28, 28), cmap = 'Blues')
plt.title(f"predicted: {y_pred[p]} | True: {y_test[p]}")
plt.show()

from sklearn.metrics import *
print(f1_score(y_pred, y_test.to('cpu'), average = 'macro'))
CM = confusion_matrix(y_pred, y_test.to('cpu'))
sns.heatmap(CM, annot = True, fmt = '.0f')
plt.show()

features = model.forward(X_cluster.to(device), True)
# features.shape

from sklearn.decomposition import PCA
pca = PCA(n_components = 300)
new_features = pca.fit_transform(features.cpu().detach().numpy())
# new_features.shape

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters = 10)
clusters = kmeans.fit_predict(new_features)
# clusters[:9]

imgs = X_cluster[clusters == 9]
for i in range(12):
    plt.subplot(3, 4, i + 1)
    plt.imshow(imgs[i].reshape(28, 28))
    plt.xticks([])
    plt.yticks([])
plt.show()