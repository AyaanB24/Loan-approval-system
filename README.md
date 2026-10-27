# LoanLens – ML-Based Loan Approval Prediction System

## 1. Title of the Activity
**Design and Development of a Machine Learning-Based Loan Approval Prediction System using Python and Streamlit**

## 2. Objective of the Activity
The objective of this activity is to develop a web-based application that uses machine learning to predict loan approval outcomes based on applicant information. The system provides an interactive interface for entering applicant details, viewing predictions, exploring datasets, and evaluating model performance.

## 3. Software, Tools, and Technologies Used

- **Programming Language:** Python
- **Frontend and Web Framework:** Streamlit
- **Machine Learning Library:** Scikit-learn
- **Data Processing:** Pandas, NumPy
- **Model Serialization:** Joblib
- **Data Visualization:** Matplotlib and other visualization libraries used in the application
- **Development Environment:** Visual Studio Code
- **Version Control:** Git and GitHub
- **Deployment Platform:** Streamlit Community Cloud

## 4. Step-by-Step Instructions for Execution

### Step 1: Install Python
Install Python on the system and verify the installation using:

```bash
python --version
```

### Step 2: Clone or Download the Project
Download the project repository from GitHub or clone it using:

```bash
git clone <your-github-repository-url>
cd <your-project-folder>
```

### Step 3: Create a Virtual Environment

```bash
python -m venv .venv
```

Activate the environment on Windows:

```powershell
.venv\Scripts\activate
```

### Step 4: Install Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

### Step 5: Train the Machine Learning Model

Run the training script:

```bash
python train_model.py
```

The script trains the configured machine learning model using the available dataset and saves the trained model if the training process completes successfully.

### Step 6: Run the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

The application opens in the default browser, usually at:

`http://localhost:8501`

### Step 7: Explore the Application

The application provides the following screens, depending on the implemented features:

1. **Overview Dashboard:** Displays a summary of the dataset and model performance.
2. **Predict Loan Approval:** Accepts applicant details and generates a prediction.
3. **Model Evaluation:** Displays evaluation metrics and visualizations.
4. **Dataset Explorer:** Allows users to inspect the available dataset.
5. **About and Limitations:** Provides project information and explains its limitations.

### Step 8: Deploy the Application

1. Push the project files to a GitHub repository.
2. Visit [Streamlit Community Cloud](https://share.streamlit.io/).
3. Sign in using GitHub.
4. Create a new app and select the project repository.
5. Select the `main` branch and set `app.py` as the main file.
6. Click **Deploy** and wait for the application to start.

## 5. Sample Input and Output

### Sample Input

The following values are illustrative examples of applicant information:

| Input Feature | Sample Value |
|---|---|
| Applicant Income | 5,000 |
| Co-applicant Income | 2,000 |
| Loan Amount | 150 |
| Loan Term | 360 months |
| Credit History | 1 |
| Education | Graduate |
| Self Employed | No |
| Property Area | Urban |

### Sample Output

The application processes the entered information and displays the predicted loan approval outcome according to the trained model.

- **Prediction:** Loan Approved or Loan Rejected
- **Probability Scores:** Displayed if supported by the implemented model and interface.

*Note: The actual prediction depends on the trained model and input data. The sample values above do not guarantee approval.*

### Screenshots

Capture and insert screenshots of the following screens after running the application:

- Overview Dashboard
- Loan Prediction Form and Result
- Model Evaluation
- Dataset Explorer
- About and Limitations

## 6. Results

The LoanLens application demonstrates the integration of machine learning with an interactive web interface. Users can enter applicant details, obtain model-generated predictions, inspect the dataset, and review model evaluation results. The application can be executed locally and deployed online through Streamlit Community Cloud.

## 7. Conclusion

The activity demonstrates how Python, Scikit-learn, Pandas, and Streamlit can be combined to develop a machine-learning-based loan approval prediction prototype. It provides practical experience in data processing, model training, model evaluation, and web application deployment.

The current application is intended for educational and demonstration purposes. Its predictions should not be used for actual lending decisions because the demo dataset and model have not been validated for real-world banking applications.
