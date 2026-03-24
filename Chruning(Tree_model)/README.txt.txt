Customer Churn Prediction App (Streamlit)

--------------------------------------------------
Overview
--------------------------------------------------
This project is a Streamlit-based web application
that predicts customer churn using multiple
tree-based machine learning models.

Users can input customer details through an
interactive interface and receive:
- Churn prediction (0 = No, 1 = Yes)
- Probability of churn

--------------------------------------------------
Features
--------------------------------------------------
- Interactive UI using Streamlit
- Preprocessing pipeline integration
- Multiple trained tree-based models
- Displays prediction + probability

--------------------------------------------------
Model Details
--------------------------------------------------
The application uses a saved pipeline (pipeline.pkl)
which contains:
- Preprocessor (data cleaning & transformation)
- Multiple tree-based models

--------------------------------------------------
Project Structure
--------------------------------------------------
app.py                -> Main Streamlit app
preprocessing.py      -> Data preprocessing logic
pipeline.pkl          -> Saved pipeline (models + preprocessor)
requirements.txt      -> Dependencies
README.txt            -> Documentation

--------------------------------------------------
Installation
--------------------------------------------------
1. Clone the repository:
   git clone <your-repo-link>
   cd <repo-folder>

2. (Optional) Create virtual environment:
   python -m venv venv
   source venv/bin/activate      (Linux/Mac)
   venv\Scripts\activate         (Windows)

3. Install dependencies:
   pip install -r requirements.txt

--------------------------------------------------
Running the App
--------------------------------------------------
Run the Streamlit app:
   streamlit run app.py

Open in browser:
   http://localhost:8501

--------------------------------------------------
Input Features
--------------------------------------------------
- Gender
- Senior Citizen
- Partner
- Dependents
- Tenure
- Phone Service
- Multiple Lines
- Internet Service
- Online Security
- Online Backup
- Device Protection
- Tech Support
- Streaming TV
- Streaming Movies
- Contract
- Paperless Billing
- Payment Method
- Monthly Charges
- Total Charges

--------------------------------------------------
Output
--------------------------------------------------
For each model:
- Prediction (0 or 1)
- Probability of churn

--------------------------------------------------
Notes
--------------------------------------------------
- Ensure pipeline.pkl is present in the root directory
- Models must support predict() and predict_proba()
- Preprocessing should match training pipeline