import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

def train(model, train_dataloader, optimizer, loss_function, epochs, device, log=True):
    """
    Training loop for classification models (stolen from Coding Lab 2).
    The classification is chosen by taking `argmax` of the final prediction.
    """
    model.train()  # Set the model to training mode
    for epoch in range(epochs):
        total_loss = 0
        correct_pred = 0
        total_pred = 0

        for i, (datasets, labels) in enumerate(train_dataloader):
            # Move tensors to GPU so compatible with model
            datasets, labels = datasets.to(device), labels.to(device)

            # Forward pass
            outputs = model(datasets)

            # Clear gradients before performing backward pass
            optimizer.zero_grad()
            # Calculate loss based on model predictions
            loss = loss_function(outputs, labels)
            # Backpropagate and update model parameters
            loss.backward()
            optimizer.step()

            # multiply loss by total nos. of samples in batch
            total_loss += loss.item()*datasets.size(0)

            # Calculate accuracy
            # Get predicted class by taking the argmax (maybe use softmax here!)
            predicted = torch.argmax(outputs, dim=-1)  
            correct_pred += (predicted == labels).sum().item()  # Count correct predictions
            total_pred += labels.size(0) # Count total predictions

        # Compute metrics
        total_epoch_loss = total_loss / total_pred
        epoch_accuracy = correct_pred / total_pred
        if log:
            print(f"Epoch {epoch + 1}, Loss: {total_epoch_loss}, Accuracy: {epoch_accuracy:.4f}")


def evaluate(model, test_dataloader, loss_function, device):
    """
    Evaluation loop for classification models (stolen from Coding Lab 2).
    The classification is chosen by taking `argmax` of the final prediction.
    
    This could be changed to something else, like `torch.nn.SoftMax`.
    """
    # Evaluate model performance on the test dataset
    model.eval()

    test_loss = 0
    correct_pred = 0
    total_pred = 0
    # Disable gradient calculations when in inference mode
    with torch.no_grad():
        for datasets, labels in test_dataloader:
            datasets, labels = datasets.to(device), labels.to(device)

            outputs = model(datasets)

            loss = loss_function(outputs, labels)

            test_loss += loss.item()*datasets.size(0)

            predicted = torch.argmax(outputs, dim=1)

            # tally the number of correct predictions
            correct_pred += torch.sum(predicted == labels).item()

            # tally the total number of predictions
            total_pred += labels.size(0)

    # Compute average loss and accuracy
    test_loss /= total_pred
    test_acc = correct_pred / total_pred
    return test_loss, test_acc


def run(Classifier, Optimizer, LossFunction, epochs, train_dataloader, test_dataloader, device, model_args = {}, optimizer_args = {}, loss_args = {}, log=False):
    model = Classifier(**model_args).to(device)
    optimizer = Optimizer(model.parameters(),**optimizer_args)
    loss_function = LossFunction(**loss_args)
    train(
        model,
        train_dataloader,
        optimizer,
        loss_function,
        epochs,
        device,
        log=log
    )
    loss, acc = evaluate(model, test_dataloader, loss_function, device)
    return loss, acc


def benchmark(Classifier, Optimizer, LossFunction, epochs, train_dataloader, test_dataloader, device, model_args = {}, optimizer_args = {}, log=False, runs=20):
    losses = []
    accs = []

    for i in range(runs):
        print(f"Starting run {i+1}...")
        loss, acc = run(Classifier, Optimizer, LossFunction, epochs, train_dataloader, test_dataloader, device, model_args = {}, optimizer_args = {}, log=False)
        print(f"Run {i+1}: loss {loss} acc {acc}")
        losses.append(loss)
        accs.append(acc)

    return np.array(losses), np.array(accs)


def show_dataset(dataset, return_figure=True):
    fig, axes = plt.subplots(3,2)

    directions = ["x", "y", "z"]

    value_type = ['acceleration', 'rotation']

    for i, direction_label in enumerate(directions):
        for j, value_label in enumerate(value_type):
            axes[i,j].plot(dataset[...,:,3*j+i])
            if i == 0:
                axes[i,j].set_title(value_label)
                
            if j == 1:
                axes[i,j].text(1, 0.5,direction_label, size=12, rotation=270, transform=axes[i,j].transAxes)
    if return_figure:
        return fig
