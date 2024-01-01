import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
from torch.optim import Adam
import copy
import time
import sys
import torch
import torch.nn as nn
import torchvision.models
import models

# # Define your ResNet class
# import torchvision.models
# class ResNet(nn.Module):
#     def __init__(self, model='resnet18', n_channels=4, n_filters=64, n_classes=2, kernel_size=3, stride=1, padding=1):
#         super().__init__()
#         self.n_classes = n_classes
#         models_dict = {
#             'resnet18': torchvision.models.resnet18,
#             'resnet34': torchvision.models.resnet34,
#             'resnet50': torchvision.models.resnet50,
#             'resnet101': torchvision.models.resnet101,
#             'resnet152': torchvision.models.resnet152
#         }
#         self.base_model = models_dict[model](pretrained=True)
#         self._feature_vector_dimension = self.base_model.fc.in_features
#         self.base_model.conv1 = nn.Conv2d(n_channels, n_filters, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
#         self.base_model = nn.Sequential(*list(self.base_model.children())[:-1]) # Remove the final fully connected layer
#         self.fc = nn.Linear(self._feature_vector_dimension, n_classes)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         x = self.base_model(x)
#         features = x.view(x.size(0), -1)
#         x = self.fc(features)
#         return x



def train(model, images, annotations, out_model_path, train_frac=0.7, batch_size=32, n_epochs=40):
	model_best = None

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model = model.to(device)

	# shuffle
	indices = np.random.choice(len(images), len(images), replace=False)
	np.savetxt(out_model_path[:out_model_path.rfind('/')] + '/indices_r34_b32.csv', indices[int(len(images)*0.7):], delimiter=',')
	data = images[indices,:,:,:]
	label = annotations[indices]

	# make images 0-1 if they are not already
	if data.dtype == np.uint8:
		data = data.astype(np.float32)/255.0 # convert to 0-1 if uint8 input

	print('splitting with ' + str(round(train_frac,1)) + ':' + str(round(1-train_frac,1)) + ' train:test split')
	# Split the data into train, validation, and test sets
	X_train, X_val = np.split(data, [int(train_frac * len(data))])
	y_train, y_val = np.split(label, [int(train_frac * len(label))])

	# Create TensorDatasets for train, validation and test
	train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
	val_dataset = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))

	train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=4)
	val_dataloader = DataLoader(val_dataset, batch_size, shuffle=False, num_workers=4)

	# initialize stats
	best_validation_loss = np.inf

	# Define the loss function and optimizer
	# can add weight parameter to differently weight classes; can also add label_smoothing to avoid overfitting
	criterion = nn.CrossEntropyLoss()
	optimizer = Adam(model.parameters(), lr=1e-3)

	# initialize loss df
	loss_df = pd.DataFrame(columns=['running loss', 'validation loss'])
	# Training loops
	for epoch in range(n_epochs):
		print(str(epoch))
		running_loss = 0.0
		model.train()
		# by batch size?
		for inputs, labels in train_dataloader:
			inputs = inputs.float().to(device)
			labels = labels.float().to(device)
			
			# Forward pass
			outputs = model(inputs)
			labels = labels.to(torch.long)
			loss = criterion(outputs, labels)

			# Backward and optimize
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			running_loss += loss.item()

		# Compute the validation performance
		validation_loss = evaluate_model(model, val_dataloader, criterion, device)
		if validation_loss < best_validation_loss:
			best_validation_loss = validation_loss
			model_best = copy.deepcopy(model)
		# save running_loss and validation_loss
		new_loss = pd.DataFrame({'running loss': running_loss, 'validation loss': validation_loss}, index=[0])
		print('Epoch ' + str(epoch) + ': running loss - '); print(running_loss); print('; validation loss - '); print(validation_loss)
		loss_df = pd.concat([loss_df, new_loss], ignore_index=True)
		print("\a"); time.sleep(0.5); print("\a"); time.sleep(0.5); print("\a"); time.sleep(0.5); print("\a"); time.sleep(0.5); 


	loss_df.to_csv(out_model_path[:out_model_path.rfind('/')] + '/loss_df_r34_b32.csv')
	# training complete
	print('saving the best model to ' + out_model_path)
	torch.save(model_best, out_model_path)


def evaluate_model(model, dataloader, criterion, device):
	model.eval()

	total_loss = 0.0
	with torch.no_grad():
		for inputs, labels in dataloader:
			inputs = inputs.float().to(device)
			labels = labels.float().to(device)

			# Forward pass
			outputs = model(inputs)
			labels = labels.to(torch.long)
			loss = criterion(outputs, labels)

			total_loss += loss.item()
	return total_loss



# input paths
ims_path = 'combined_images_parasite_and_non-parasite.npy' # path to images for training
label_path = 'combined_ann_parasite_and_non-parasite.csv' # path to their respective annotations

images = np.load(ims_path)
labels_df = pd.read_csv(label_path, index_col='index')
labels = labels_df['annotation'].values

# model architecture
model_specs = {'model_name':'resnet34','n_channels':4,'n_filters':64,'n_classes':2,'kernel_size':3,'stride':1,'padding':1, 'batch_size':32}

n_classes_derived = labels_df['annotation'].nunique() # number of unique annotation classes in dataset
model = models.ResNet(model=model_specs['model_name'],n_channels=model_specs['n_channels'],n_filters=model_specs['n_filters'], n_classes=n_classes_derived,kernel_size=model_specs['kernel_size'],stride=model_specs['stride'],padding=model_specs['padding'])

train_frac = 0.7
n_epochs = 2
model_out_path = './'+ model_specs['model_name'] + '_' + str(time.time()) + '.pt'
train(model, images, labels, model_out_path, train_frac, model_specs['batch_size'], n_epochs)