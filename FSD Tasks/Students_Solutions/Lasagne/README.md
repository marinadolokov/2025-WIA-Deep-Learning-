## 1. Group Information

- **Group Name**: Lasagna  
- **Members**: Moritz Handrejk, Tim Rothe, Helmi Ghliss, Malak Mrini
- **Project Title**: Deep Learning models and synthetic data generation for vehicle brake condition classification  
- **Short Description**: This repository investigates braking condition classification using Deep Learning models with extensive hyperparameter tuning and synthetic data generation using cGANs and RNNs with its application in classification training. 

---

## 2. Setup & Execution Instructions

### Environment Setup (via `pip`)

Please install the python package requirements via:

```python
pip install -r requirements.txt
```

For GPU acceleration make sure you install the right version of `pytorch`.
Especially for training in Task 2 it helps to have GPU acceleration.

**For interactive analysis in `augmentation.ipynb`,`benchmark.ipynb` and `data_analysis.ipynb` it is recommended to use `JupyterLab`, since parts of the dataset view is implemented using `ipywidgets`!**

We only worked in a Linux environment (including WSL2), therefore we did not make sure that this will run on Windows as well.

It was also successfully tried to run `pytorch` with AMD GPU acceleration. 
A Dockerfile for the environment is in this folder, though running `pip install -r requirements.txt` might resolve additional dependencies. 
It will automatically run a `jupyter-lab` instance on start. 
If you want to run this, please follow [AMD's installation guide](https://rocm.docs.amd.com/en/latest/) first.

### Running the Code

The project has a straightforward structure. 
As a first task (folder `task1`) we tried to find useful parameter combinations for braking classifiers based on different neural network architectures (`cnn`,`fcn` and `lstm`).
The notebook for the parameter search for every architecture can be found in its folder, together with a python file `model.py` which exports the chosen architecture. 
Finally, to compare the performance of different models a jupyter notebook called `benchmark.ipynb` is supplied.
This allows to compare the models under the same environment using random-splitting and feedback from the test dataset.

The second task (folder `task2`) is structured in the same manner.
For each architecture (`gan` and `rnn`) the folders show how the training of the networks is implemented and reflect the training process in the report. 
The python files `model.py` export the modules.
Finally, the results get compared in `augmentation.ipynb` by importing the same modules from Task 2 and filling these with trained parameters.

As a small bonus, a notebook called `data_analysis.ipynb` shows the general structure of the training dataset.

### Viewing Results

#### Task 1

The following steps can be taken:

1. Read the [report](results/report.pdf) located in `results`. It is already full of exported diagrams, which show the results.
2. Run `benchmark.ipynb` for specific performance tests of the implemented models under similar conditions.
3. Finally, run the parameter search yourself using the notebooks in their respective folders (`cnn`, `fcn` and `lstm`).

#### Task 2

The following steps can be taken:

1. Read the [report](results/report.pdf) located in `results`. It is already full of exported diagrams, which show the results.
2. Train the GAN by running `gan.ipynb`. This notebook exports parameters to `models/generator-gan.pt` of the directory of Task 2. An already trained model is provided.
3. Train the RNN by running `rnn-teacher-forcing-embedding.ipynb`. This notebook exports parameters to `models/generator-rnn.pt` of the directory of Task 2. An already trained model is provided.
4. Use the parameters of each model and compare the outputs by running `augmentation.ipynb`.