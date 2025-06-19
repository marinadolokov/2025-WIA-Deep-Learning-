import matplotlib.pyplot as plt
import torch
import numpy as np

REAL_LABEL = 1
FAKE_LABEL = 0

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

def generate_mean_by_class(dataset):
    class_list = []
    
    for i in range(12):
        selected_data = np.array(list(dataset.query(f'label == {i}')['sensor_data']), dtype=np.float32)
        class_list.append(np.mean(selected_data, axis=0))

    return np.array(class_list)


def normalize(tensor):
    with torch.no_grad():
        return (tensor - tensor.min(dim=-1, keepdims=True)[0] ) / (tensor.max(dim=-1, keepdims=True)[0] - tensor.min(dim=-1, keepdims=True)[0])


def evaluate_discriminator(generator, discriminator, test_dataloader, device):
    # set generator and discriminator evaluation
    generator.eval()
    discriminator.eval()

    generator = generator.to(device)
    discriminator = discriminator.to(device)

    total_pred = 0
    correct_pred = 0
    fake_pred = 0
    real_pred = 0

    with torch.no_grad():
        for i, (datasets, labels) in enumerate(test_dataloader):
            batch_size = datasets.shape[0]

            ### REAL DATA
            # move to GPU
            labels = labels.to(device)
            datasets = datasets.to(device)
            output = discriminator(datasets, labels)

            # compute correct predictions
            real_pred += torch.sum(output.round() == REAL_LABEL)
            correct_pred += torch.sum(output.round() == REAL_LABEL)
            total_pred += batch_size

            ###  FAKE DATA
            # from noise generate fake data
            noise = generator.sample(batch_size=batch_size, device=device)
            fake_data = generator(noise, labels)

            # evaluate fake data
            output = discriminator(fake_data, labels)

            # compute correct prediction#
            # print(output)
            fake_pred += torch.sum(output.round() == FAKE_LABEL)
            correct_pred += torch.sum(output.round() == FAKE_LABEL)
            total_pred += batch_size
            
    accuracy = correct_pred / total_pred
    accuracy_real = real_pred / total_pred * 2
    accuracy_fake = fake_pred / total_pred * 2
    return accuracy.item(), (accuracy_real.item(), accuracy_fake.item())
