import torch
import numpy as np
import time
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
from torch.optim import Adam
import copy

def generate_predictions_and_features(model,images,batch_size, verbose=False):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    if images.dtype == np.uint8:
        images = images.astype(np.float32)/255.0 # convert to 0-1 if uint8 input

    # build dataset
    dataset = TensorDataset(
        torch.from_numpy(images), 
        torch.from_numpy(np.ones(images.shape[0]))
        )

    # dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # run inference 
    all_features = []
    all_predictions = []
    t0 = time.time()

    for k, (images, labels) in enumerate(dataloader):

        images = images.float().to(device)

        predictions, features = model.get_predictions_and_features(images)
        predictions = predictions.detach().cpu().numpy()
        features = features.detach().cpu().numpy().squeeze()

        all_predictions.append(predictions)
        all_features.append(features)

    predictions = np.vstack(all_predictions)
    features = np.vstack(all_features)

    if verbose:
        print('running inference on ' + str(predictions.shape[0]) + ' images took ' + str(time.time()-t0) + ' s')

    '''
    # Plot
    print(predictions.shape)
    import matplotlib.pyplot as plt
    plt.clf()
    plt.hist(predictions, bins=100, range=[0,1])
    plt.xlabel('Output')
    plt.ylabel('Count')
    plt.title('Histogram of Outputs')
    plt.savefig('histograms_' + str(0)  + '.png', dpi=300)
    '''

    return predictions, features


def train_model(model,images,annotations,batch_size,n_epochs,model_name,annotations_dict,reset=False,caller=None):

    model_best = None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    if reset:
        print('reset model parameters')
        model.reset_parameters()

    # make images 0-1 if they are not already
    if images.dtype == np.uint8:
        images = images.astype(np.float32)/255.0 # convert to 0-1 if uint8 input

    # shuffle
    indices = np.random.choice(len(images), len(images), replace=False)
    data = images[indices,:,:,:]
    label = annotations[indices]

    # Split the data into train, validation, and test sets
    X_train, X_val = np.split(data, [int(.7 * len(data))])
    y_train, y_val = np.split(label, [int(.7 * len(label))])

    # Create TensorDatasets for train, validation and test
    train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_dataset = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))

    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size, shuffle=False, num_workers=4)

    # initialize stats
    best_validation_loss = np.inf
    TP_accum = 0
    TN_accum = 0
    FP_accum = 0
    FN_accum = 0

    # Define the loss function and optimizer
    # criterion = nn.BCEWithLogitsLoss()
    # can add weight parameter to differently weight classes; can also add label_smoothing to avoid overfitting
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=1e-3)

    # Training loops
    for epoch in range(n_epochs):

        if caller:
            if caller.stop_requested:
                if model_best is not None:
                    caller.model = copy.deepcopy(model_best)
                    caller.model_loaded = True
                    print('saving the model to ' + model_name + '.pt')
                    torch.save(model, model_name + '.pt')
                caller.signal_training_complete.emit()
                return

        running_loss = 0.0

        model.train()
        
        for inputs, labels in train_dataloader:
            
            # inputs = inputs.float().cuda()
            # labels = labels.cuda()
            inputs = inputs.float().to(device)
            labels = labels.float().to(device)
            
            # Forward pass
            outputs = model(inputs)
            # outputs = outputs.view(-1) # maybe this is just for binary
            labels = labels.to(torch.long)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Squash the outputs using sigmoid function for binary
            # outputs = torch.sigmoid(outputs) 
            # Squash the outputs using the softmax function for multiclass
            outputs = torch.softmax(outputs, dim = 1)

            # Threshold the outputs to obtain the predictions
            # predictions = (outputs > 0.5).float()
            # predictions = predictions.view(-1)
            # Use argmax to get the class index with the highest probability
            predictions = torch.argmax(outputs, dim = 1)
            # annotations_values = [key for key in annotations_dict.keys() if key != -1]
            # predictions = torch.tensor([annotations_values[i] for i in prediction_indices]).to(device)

            # TODO: fix TP, etc. for multiclass
            TP = ((predictions == 1) & (labels == 1)).sum().item()
            TN = ((predictions == 0) & (labels == 0)).sum().item()
            FP = ((predictions == 1) & (labels == 0)).sum().item()
            FN = ((predictions == 0) & (labels == 1)).sum().item()

            # Accumulate the values
            TP_accum += TP
            TN_accum += TN
            FP_accum += FP
            FN_accum += FN

            running_loss += loss.item()

        if FP_accum + TN_accum > 0:
            FPR = FP_accum / (FP_accum + TN_accum)
        else:
            FPR = np.NAN

        if FN_accum + TP_accum > 0:
            FNR = FN_accum / (FN_accum + TP_accum)
        else:
            FNR = np.NAN
        print('Epoch {}: FPR: {:.4f} FNR: {:.4f}'.format(epoch+1, FPR, FNR))

        # Compute the validation performance
        validation_loss = evaluate_model(model, val_dataloader, criterion, device, annotations_dict)
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            # torch.save(model.state_dict(), 'best_model.pt')
            # torch.save(model, model_name + '.pt')
            # caller.model = model # read from disk instead
            model_best = copy.deepcopy(model)

        if caller:
            caller.signal_progress.emit(100*(epoch+1)/n_epochs)
            caller.signal_update_loss.emit(epoch,running_loss,validation_loss)
    
    # training complete
    if caller:
        if model_best is not None:
            caller.model = copy.deepcopy(model_best)
            caller.model_loaded = True
            print('saving the model to ' + model_name + '.pt')
            torch.save(model_best, model_name + '.pt')
        caller.signal_training_complete.emit()

def evaluate_model(model, dataloader, criterion, device, annotations_dict):

    TP_accum = 0
    TN_accum = 0
    FP_accum = 0
    FN_accum = 0

    model.eval()

    total_loss = 0.0
    start_time = time.time()
    with torch.no_grad():
        for inputs, labels in dataloader:
            # inputs = inputs.float().cuda()
            # labels = labels.cuda()
            inputs = inputs.float().to(device)
            labels = labels.float().to(device)

            # Forward pass
            outputs = model(inputs)
            # outputs = outputs.view(-1) # maybe this is just for binary
            labels = labels.to(torch.long)
            loss = criterion(outputs, labels)

            total_loss += loss.item()

            # Squash the outputs using sigmoid function for binary
            # outputs = torch.sigmoid(outputs) 
            # Squash the outputs using the softmax function for multiclass
            outputs = torch.softmax(outputs, dim = 1)

            # Threshold the outputs to obtain the predictions
            # predictions = (outputs > 0.5).float()
            # predictions = predictions.view(-1)
            # Use argmax to get the class index with the highest probability
            predictions = torch.argmax(outputs, dim = 1)
            # annotations_values = [key for key in annotations_dict.keys() if key != -1]
            # predictions = torch.tensor([annotations_values[i] for i in prediction_indices]).to(device)

            # TODO: fix TP, etc. for multiclass
            TP = ((predictions == 1) & (labels == 1)).sum().item()
            TN = ((predictions == 0) & (labels == 0)).sum().item()
            FP = ((predictions == 1) & (labels == 0)).sum().item()
            FN = ((predictions == 0) & (labels == 1)).sum().item()

            # Accumulate the values
            TP_accum += TP
            TN_accum += TN
            FP_accum += FP
            FN_accum += FN

    end_time = time.time()
    total_time = end_time - start_time
    num_samples = len(dataloader.dataset)
    throughput = (num_samples) / total_time
    # print('Processed {:d} samples; Throughput is {:.1f} images/s'.format(num_samples,throughput))

    if FP_accum + TN_accum > 0:
        FPR = FP_accum / (FP_accum + TN_accum)
    else:
        FPR = np.NAN

    if FN_accum + TP_accum > 0:
        FNR = FN_accum / (FN_accum + TP_accum)
    else:
        FNR = np.NAN
    print('    [validation] Loss {:.4f} FPR: {:.4f} FNR: {:.4f}'.format(total_loss, FPR, FNR))

    return total_loss
