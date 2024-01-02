import torch
import torch.nn as nn
import torchvision
from skorch import NeuralNetClassifier
import numpy as np
import pickle
import matplotlib.pyplot as plt
import time
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset, random_split, Subset
from tqdm import tqdm
import math
import multiprocessing
import torch.optim as optim


from cleanlab import Datalab

#https://docs.cleanlab.ai/stable/index.html
# lab = Datalab(data=your_dataset, label_name="column_name_of_labels")
# lab.find_issues(features=feature_embeddings, pred_probs=pred_probs)
# lab.report()  # summarize issues in dataset, how severe they are, ...

# Define your ResNet class
class ResNet(nn.Module):
    def __init__(self, model='resnet18', n_channels=4, n_filters=64, n_classes=3, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.n_classes = n_classes
        models_dict = {
            'resnet18': torchvision.models.resnet18,
            'resnet34': torchvision.models.resnet34,
            'resnet50': torchvision.models.resnet50,
            'resnet101': torchvision.models.resnet101,
            'resnet152': torchvision.models.resnet152
        }
        self.base_model = models_dict[model](pretrained=False)
        self._feature_vector_dimension = self.base_model.fc.in_features
        self.base_model.conv1 = nn.Conv2d(n_channels, n_filters, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.base_model = nn.Sequential(*list(self.base_model.children())[:-1]) # Remove the final fully connected layer
        self.fc = nn.Linear(self._feature_vector_dimension, n_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.base_model(x)
        features = x.view(x.size(0), -1)
        x = self.fc(features)
        return x

    def embeddings(self, x):
        x = self.base_model(x)
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        # x = self.linear(x)
        return x

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Method to calculate validation accuracy in each epoch
def get_test_accuracy(net, testloader):
    net.eval()
    accuracy = 0.0
    total = 0.0

    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)

            # run the model on the test set to predict labels
            outputs = net(images)

            # the label with the highest energy will be our prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            accuracy += (predicted == labels).sum().item()

    # compute the accuracy over all test images
    accuracy = 100 * accuracy / total
    return accuracy


# Method for training the model
def train(trainloader, testloader, n_epochs, patience):
    model = ResNet(model='resnet34', n_channels=4, n_filters=64, n_classes=3, kernel_size=3, stride=1, padding=1)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters())

    model = model.to(device)

    best_test_accuracy = 0.0

    for epoch in range(n_epochs):  # loop over the dataset multiple times
        start_epoch = time.time()
        running_loss = 0.0

        for _, data in enumerate(trainloader):
            # get the inputs; data is a dict of {"image": images, "label": labels}

            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.detach().cpu().item()

        # Get accuracy on the test set
        accuracy = get_test_accuracy(model, testloader)

        if accuracy > best_test_accuracy:
            best_epoch = epoch

        # Condition for early stopping
        if epoch - best_epoch > patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

        end_epoch = time.time()

        print(
            f"epoch: {epoch + 1} loss: {running_loss / len(trainloader):.3f} test acc: {accuracy:.3f} time_taken: {end_epoch - start_epoch:.3f}"
        )
    return model


# Method for computing out-of-sample embeddings
def compute_embeddings(model, testloader):
    embeddings_list = []

    with torch.no_grad():
        for data in tqdm(testloader):
            images, labels = data[0].to(device), data[1].to(device)

            embeddings = model.embeddings(images)
            embeddings_list.append(embeddings.cpu())

    return torch.vstack(embeddings_list)


# Method for computing out-of-sample predicted probabilities
def compute_pred_probs(model, testloader):
    pred_probs_list = []

    with torch.no_grad():
        for data in tqdm(testloader):
            images, labels = data[0].to(device), data[1].to(device)

            outputs = model(images)
            pred_probs_list.append(outputs.cpu())

    return torch.vstack(pred_probs_list)


# settings
fraction = 0.0005
n_epochs = 1  # Number of epochs to train model for. Set to a small value here for quick runtime, you should use a larger value in practice.

# Load the data
parasite_images = np.load('data/parasite.npy')
negative_images = np.load('data/negative.npy')
unsure_images = np.load('data/unsure.npy')

# Create labels
parasite_labels = np.ones(len(parasite_images))  # Label 1 for parasites
negative_labels = np.zeros(len(negative_images))  # Label 0 for non-parasites
unsure_labels = 2*np.ones(len(unsure_images))  # Label 0 for non-parasites

# Combine the datasets
images = np.concatenate((parasite_images, negative_images, unsure_images), axis=0)
labels = np.concatenate((parasite_labels, negative_labels, unsure_labels), axis=0)
labels = labels.astype("int64")

# Sample the data
sample_indices = np.random.choice(len(images), size=int(fraction*len(images)), replace=False)
# sample_indices = np.sort(sample_indices)
images = images[sample_indices]
labels = labels[sample_indices]

images = images/255.0
images = images.astype("float32")

print(len(images))

# # Convert to torch tensors
# images = torch.tensor(images, dtype=torch.float32)
# labels = torch.tensor(labels, dtype=torch.long)

# # Create a dataset
# dataset = TensorDataset(images, labels)

# Create a dictionary from the images and labels, Create a Hugging Face dataset from the dictionary
data_dict = {
    'images': images.tolist(),  # Convert numpy arrays to lists
    'labels': labels.tolist()
}
from datasets import Dataset
dataset = Dataset.from_dict(data_dict)

transformed_dataset = dataset.with_format("torch")
torch_dataset = TensorDataset(transformed_dataset["images"], transformed_dataset["labels"])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Save the sampled images
id_ = str(time.time())
np.save('sampled_images_'+id_+'.npy', images)
labels_df = pd.DataFrame(labels, columns=['annotation'])
labels_df.to_csv('sampled_labels_'+id_+'.csv', index=True)


### 4. Prepare the dataset for K-fold cross-validation
K = 5  # Number of cross-validation folds. Set to small value here to ensure quick runtimes, we recommend 5 or 10 in practice for more accurate estimates.
patience = 3  # Parameter for early stopping. If the validation accuracy does not improve for this many epochs, training will stop.
train_batch_size = 32  # Batch size for training
test_batch_size = 1024  # Batch size for testing
num_workers = multiprocessing.cpu_count()  # Number of workers for data loaders

# Create k splits of the dataset
from sklearn.model_selection import StratifiedKFold
kfold = StratifiedKFold(n_splits=K, shuffle=True, random_state=0)
splits = kfold.split(transformed_dataset, transformed_dataset["labels"])

train_id_list, test_id_list = [], []

for fold, (train_ids, test_ids) in enumerate(splits):
# for fold, (train_ids, test_ids) in enumerate(kfold.split(torch.zeros(len(labels)), labels.numpy())):
    train_id_list.append(train_ids)
    test_id_list.append(test_ids)


### 5. Compute out-of-sample predicted probabilities and feature embeddings
pred_probs_list, embeddings_list = [], []
embeddings_model = None

for i in range(K):
    print(f"\nTraining on fold: {i+1} ...")

    # Create train and test sets and corresponding dataloaders
    trainset = Subset(torch_dataset, train_id_list[i])
    testset = Subset(torch_dataset, test_id_list[i])

    trainloader = DataLoader(
        trainset,
        batch_size=train_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    testloader = DataLoader(
        testset, batch_size=test_batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )

    # Train model
    model = train(trainloader, testloader, n_epochs, patience)
    if embeddings_model is None:
        embeddings_model = model

    # Compute out-of-sample embeddings
    print("Computing feature embeddings ...")
    fold_embeddings = compute_embeddings(embeddings_model, testloader)
    embeddings_list.append(fold_embeddings)

    print("Computing predicted probabilities ...")
    # Compute out-of-sample predicted probabilities
    fold_pred_probs = compute_pred_probs(model, testloader)
    pred_probs_list.append(fold_pred_probs)

print("Finished Training")


# Combine embeddings and predicted probabilities from each fold
features = torch.vstack(embeddings_list).numpy()

logits = torch.vstack(pred_probs_list)
pred_probs = nn.Softmax(dim=1)(logits).numpy()

indices = np.hstack(test_id_list)
dataset = dataset.select(indices)

print(len(indices))

# print(type(dataset))
# print(type(features))
# print(type(pred_probs))
dataset.save_to_disk('dataset')
np.save('features.npy', features)
np.save('pred_probs.npy', pred_probs)

### 7. Use cleanlab to find issues
possible_issue_types = {
    "label": {}, "outlier": {},
    "near_duplicate": {}, "non_iid": {}
}

lab = Datalab(data=dataset, label_name="labels", image_key="images")
# print(lab.find_issues)
lab.find_issues(features=features, pred_probs=pred_probs, issue_types=possible_issue_types)
# print(lab.report())
lab.report()

# Save the lab object to a file
import pickle
with open('lab.pkl', 'wb') as file:
    pickle.dump(lab, file)


'''
# Convert numpy arrays to torch tensors and scale to [0, 1]
images = torch.tensor(images, dtype=torch.float32) / 255.0  # Scale to [0, 1]



# model
model = 'resnet34'

# Define the neural network using skorch
net = NeuralNetClassifier(
    module=ResNet,
    module__model=model,  # Choose the ResNet model you want
    module__n_channels=4,
    criterion=nn.CrossEntropyLoss,
    optimizer=torch.optim.Adam,
    lr=0.001,
    batch_size=32,
    max_epochs=5,
    iterator_train__shuffle=True,
    device='cuda' if torch.cuda.is_available() else 'cpu'  # Use GPU if available
)

# load device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# out of sample
from sklearn.model_selection import cross_val_predict
num_crossval_folds = 5
pred_probs = cross_val_predict(net, images, labels,
                               cv=num_crossval_folds,
                               method='predict_proba')

from sklearn.metrics import accuracy_score
predicted_labels = pred_probs.argmax(axis=1)
acc = accuracy_score(label, predicted_labels)
print(f"Cross-validated estimate of accuracy on held-out data: {acc}")


from cleanlab.filter import find_label_issues
ranked_label_issues = find_label_issues(y, pred_probs, return_indices_ranked_by="self_confidence")
print(f"Cleanlab found {len(ranked_label_issues)} label issues.")
print("Here are the indices of the top 15 most likely label errors:\n"
      f"{ranked_label_issues[:15]}")

import matplotlib.pyplot as plt

def plot_examples(id_iter, nrows=1, ncols=1):
    for count, id in enumerate(id_iter):

        frame = images[id]
        frame = frame.transpose(1,2,0)
        img_fluorescence = frame[:,:,[2,1,0]]
        img_dpc = frame[:,:,3]
        img_dpc = np.dstack([img_dpc,img_dpc,img_dpc])
        img_overlay = 0.64*img_fluorescence + 0.36*img_dpc

        plt.subplot(nrows, ncols, count + 1)
        plt.imshow(img_overlay)
        plt.title(f"id: {id} \n label: {label[id]}")
        plt.axis("off")

    plt.tight_layout(h_pad=2.0)
    plt.show()

plot_examples(ranked_label_issues[range(15)], 3, 5)

print(ranked_label_issues)
np.save(str(len(ranked_label_issues)) + ' possible label issues.npy',images[ranked_label_issues])
'''