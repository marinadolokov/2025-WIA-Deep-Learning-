# Group Project Submission – WIA Deep Learning, Neural Networks and Applications 

Welcome to the group project workspace for WIA Deep Learning, Neural Networks & Applications.

Each group should submit their work in a dedicated subfolder of this repository using the following format:
```
groupname (for example "HalloLeute")/ 
├── README.md
├── code/
├── data/
├── environment.yml / requirements.txt
└── results/
```

- Please make sure to upload only code files and not data files, as otherwise the repository will become too large. However, if you have trained and saved a final model, you may include it in the repository.

- Please add the final project report as a PDF file to the results folder. The final report should be approximately 15 pages long. However, it may be longer if you consider it necessary. Please indicate who worked on which part. Present your approach and your results. The report serves as documentation and will also be forwarded to the professor. It is one of the requirements for passing the seminar.

- Please ensure your `README.md` contains the following sections.

---

## 1. Group Information

- **Group Name**: GroupXX  
- **Members**: Full names   
- **Project Title**: Descriptive and concise  
- **Short Description**: Brief summary of your problem statement, approach, and goal

---

## 2. Setup & Execution Instructions

Please provide everything needed to run your code and reproduce your results.

### Environment Setup

Choose one of the following options:

<details>
<summary><strong>Option A: Conda</strong></summary>

```bash
# Create the environment
conda env create -f environment.yml

# Activate the environment
conda activate myenv
```
</details>

<details>
<summary><strong>Option B: pip</strong></summary>

```bash
# Install requirements
pip install -r requirements.txt
```
</details>

### Running the Code

Explain clearly how to execute your main script or notebook:

```bash
# Example: run main pipeline
python main.py --input data/test.csv --output results/
```

### Viewing Results

Indicate where to find results (e.g., plots, reports, model outputs):

```bash
# Example:
Open results/final_report.pdf
```

---

## 3. Evaluation Criteria

The following criteria will be used to assess your submission.

| Criterion            | Description                                               | 
|----------------------|-----------------------------------------------------------|
| Code Quality         | Is the code clean, modular, and readable?                 |                   
| Documentation        | Are the structure and logic of the project clear?         |                   
| Functionality        | Does the code run and deliver expected results?           |                   
| Innovation           | Does the project show creativity or novel approaches?     |                   
| Teamwork             | Was the project well-organized within the group?          |                   


---

## 4. License & Acknowledgments (optional)

- If your code uses third-party packages or datasets, cite them appropriately here.
- You may choose to add a license if you wish to open-source your work.

---

Thank you for your submission!
