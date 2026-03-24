from preprocessing import preprocess
import pandas as pd
import joblib

def get_user_input():
    # You can modify prompts or types as needed
    data = {}
    data['gender'] = input("Gender (Male/Female): ")
    data['SeniorCitizen'] = int(input("SeniorCitizen (0/1): "))
    data['Partner'] = input("Partner (Yes/No): ")
    data['Dependents'] = input("Dependents (Yes/No): ")
    data['tenure'] = int(input("Tenure (months): "))
    data['PhoneService'] = input("PhoneService (Yes/No): ")
    data['MultipleLines'] = input("MultipleLines (Yes/No/No phone service): ")
    data['InternetService'] = input("InternetService (DSL/Fiber optic/No): ")
    data['OnlineSecurity'] = input("OnlineSecurity (Yes/No/No internet service): ")
    data['DeviceProtection'] = input("DeviceProtection (Yes/No/No internet service): ")
    data['OnlineBackup'] = input("OnlineBackup (Yes/No/No internet service): ")
    data['TechSupport'] = input("TechSupport (Yes/No/No internet service): ")
    data['StreamingTV'] = input("StreamingTV (Yes/No/No internet service): ")
    data['StreamingMovies'] = input("StreamingMovies (Yes/No/No internet service): ")
    data['Contract'] = input("Contract (Month-to-month/One year/Two year): ")
    data['PaperlessBilling'] = input("PaperlessBilling (Yes/No): ")
    data['PaymentMethod'] = input("PaymentMethod (Electronic check/Mailed check/Bank transfer/Credit card): ")
    data['MonthlyCharges'] = float(input("MonthlyCharges: "))
    data['TotalCharges'] = float(input("TotalCharges: "))
    
    # Convert to 1-row DataFrame
    df = pd.DataFrame([data])
    return df

while(1):
    try:
        data = get_user_input()
        break
    except Exception as e:
        print(e,'Try again')

try:
    pipe = joblib.load('pipeline.pkl')

    preprocess = pipe['prepocesser']
    models = pipe['models']

    data = preprocess(data)
    # print(data.keys())
    data = data.drop('Churn',axis=1)
    print('Answer from RandomFroestClassifier-> ',models['RandomForestClassifier'].predict(data))
    print('Answer from XGBClassifier-> ',models['XGBClassifier'].predict(data))
except Exception as e:
    print('Error occured-> ',e)