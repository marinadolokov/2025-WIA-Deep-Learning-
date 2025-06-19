## 1. Group Information

- **Group Name**: HalloLeute  
- **Members**: Daniel Useche Corchuelo, Sebastian Garcia Sarmiento, Nikita Biswas, Chaeran Yoo   
- **Project Title**:  Classifying Deceleration Measurement Sequences
- **Short Description**: Classify whether a vehicle’s braking system is intact or defec-
tive using sensor data from deceleration measurements series and using networks for that. Three
different types of neural network architectures are available: a Fully Convolutional
Network (FCN), a 2D Convolutional Neural Network (CNN), and a Long Short-Term Memory
(LSTM) network. Additional architectures are also available, and techniques such as preprocessing and data augmentation.

---

## 2. Setup & Execution Instructions

### Parameters Setup

The parameters are extracted from /configs folder.<br />
The parameters description is in the file params.mb

### Environment Setup

<details>
<summary><strong>Conda</strong></summary>

```bash
# Create the environment
conda env create -f environment.yml

# Activate the environment
conda activate myenv

#Use env in jupyter notebook
python -m ipykernel install --user --name=my-env --display-name "my-env"
```
</details>


### Running the Code
The notebook is code.ipynb.<br />
-Run the Import cell.<br />
-Run loading files cell.<br />
-Depending on the choice of model, import the relevant params.<br />

```python
    #Params to use
    # from xxx_params import params
    from configs.cnn_params import params
```

All the functions are under the section Functions. <br />
To run experiments run the cell under Main section.<br />

### Viewing Results

 Comet ML is used to track our model development and training runs. First, sign up for a Comet account. Enter your API key in the API_dict (hold different users).
```python
# Example:
API_dict = {"User1": "API_user1",
            "User2": "API_user2"}
```
Also, results are save in a file named Results automatically.

## 3. VAE

The variational autoencoder is independ of the rest. To run this model just run stand alone cell under VAE section.





