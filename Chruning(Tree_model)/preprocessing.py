
def preprocess(data):
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder
    if 'customerID' in data.columns:
        data = data.drop('customerID',axis=1)
    data['gender'] = data['gender'].map({'Female':0,'Male':1})
    data['Partner'] = data['Partner'].map({'Yes':1,'No':0})
    data['Dependents'] = data['Dependents'].map({'Yes':1,'No':0})
    data['PhoneService'] = data['PhoneService'].map({'Yes':1,'No':0})
    data['MultipleLines'] = data['MultipleLines'].map({'Yes':1,'No':-1,'No phone service':0})
    data['InternetService'] = data['InternetService'].map({'DSL':1,'No':-1,'Fiber optic':0})
    data['OnlineSecurity'] = data['OnlineSecurity'].map({'Yes':1,'No':-1,'No internet service':0})
    data['DeviceProtection'] = data['DeviceProtection'].map({'Yes':1,'No':-1,'No internet service':0})
    data['OnlineBackup'] = data['OnlineBackup'].map({'Yes':1,'No':-1,'No internet service':0})
    data['TechSupport'] = data['TechSupport'].map({'Yes':1,'No':-1,'No internet service':0})
    data['StreamingTV'] = data['StreamingTV'].map({'Yes':1,'No':-1,'No internet service':0})
    data['StreamingMovies'] = data['StreamingMovies'].map({'Yes':1,'No':-1,'No internet service':0})
    data['Contract'] = data['Contract'].map({'Two year':2,'One year':1,'Month-to-month':0})
    data['PaperlessBilling'] = data['PaperlessBilling'].map({'Yes':1,'No':0})
    le = LabelEncoder()
    data['PaymentMethod'] = le.fit_transform(data['PaymentMethod'])
    data['TotalCharges'] = data['TotalCharges'].replace(' ',0.0)
    data['TotalCharges'] = data['TotalCharges'].astype('float64')
    # data['Churn'] = data['Churn'].map({'Yes':1,'No':0})
    return data

# import pandas as pd
# data = pd.read_csv('new_data.csv')
# data = 
# data = preprocess(data)

# print(data.info())