## 1. Group Information

- **Group Name**: ZLXM
- **Members**: Jingyao Zhang, Chenwei Li, Moritz Bauer, Xuliang Xu
- **Project Title**: Generative AI - breaking the secret of the breaks
- **Short Description**: In this task we tackle the problem of finding a classifier to predict the status of car breaks. To to this we use different models and also generated data from generative neural networks. 

---

### Running the Code

Since we also used open CPU-Ressources attached to collab/notebook environments, we mainly provide notebooks which can be opened at [Kaggle](https://www.kaggle.com/code), [Google Collab](https://colab.research.google.com/)... or your own computer using the jupyter extension (also intalled by requirements.txt) within your favourite enviroment (tested with visual studio code)

The notebooks expect to find files data/test.pickle and data/train.pickle.

We recommend that you run cell by cell since some code blocks are optional. The notebooks should be commented and understandable. 
If you really want to run a notebook from the command line, use: 

```bash
# Example: run LSTM modell
jupyter nbconvert --to notebook --execute --inplace ./code/LSTM/LSTM_2label.ipynb
```
Task 1: FCN, 2D CNNs vs. LSTMs Parameter Tests & Comparison

FCN:
```bash
Run All ./code/FCN/BestFCN_2Evo_withOUT_fliter.ipynb

Run All ./code/FCN/FCN_1Evo_Methods.ipynb

Run All ./code/FCN/FCN_UseTest_bestAcc.ipynb
```

2D CNN:
```bash
Run All ./code/2DCNN/2DCNN+GAN+ENSEMBLE.ipynb

Run All ./code/2DCNN/2DCNNnew.ipynb
```

LSTM:
```bash
Run All ./code/LSTM/LSTM_2label.ipynb

Run All ./code/LSTM/LSTM_multilabel.ipynb
```

Task 2: Generative AI

```bash
# prepare data and train model
Run All ./code/Generative/FSDTask2Generative.ipynb

# result analysis and plot
Run All ./code/Generative/GenerativeResultsAnalysis.ipynb
```

### Viewing Results

Our report is stored in results/final_report.pdf

```bash
# Example:
Open results/final_report.pdf
```
Task 2: Generative AI

```bash
# model results
Open ./results/models/gan_discriminator.pth, ./results/models/gan_generator.pth, ./results/models/rnn.pth, ./results/models/cgan_discriminator.pth, ./results/models/cgan_generator.pth, ./results/models/dann_discriminator.pth, ./results/models/dann_generator.pth, ./results/models/dann.pth, ./results/models/tgan_discriminator.pth, ./results/models/tgan_generator.pth

# result analysis and plot
Open ./results/sensor_comparison.png
```

---

## 3. License & Acknowledgments (optional)

- Name: jupyter, Version: 1.0.0, Summary: Jupyter metapackage. Install all the Jupyter components in one go., Home-page: http://jupyter.org, Author: Jupyter Development Team
- Name: torch, Version: 2.7.0, Summary: Tensors and Dynamic neural networks in Python with strong GPU acceleration, Home-page: https://pytorch.org/, Author: PyTorch Team
- Name: numpy, Version: 1.24.3, Summary: Fundamental package for array computing in Python, Home-page: https://www.numpy.org, Author: Travis E. Oliphant et al.
- Name: matplotlib, Version: 3.7.2, Summary: Python plotting package, Home-page: https://matplotlib.org, Author: John D. Hunter, Michael Droettboom
- Name: tqdm, Version: 4.65.0, Summary: Fast, Extensible Progress Meter, Home-page: https://tqdm.github.io
- Name: pandas, Version: 2.1.4, Summary: Powerful data structures for data analysis, time series, and statistics, Home-page: https://pandas.pydata.org
- Name: scikit-learn, Version: 1.3.0, Summary: A set of python modules for machine learning and data mining, Home-page: http://scikit-learn.org
---

Thank you for your submission!
