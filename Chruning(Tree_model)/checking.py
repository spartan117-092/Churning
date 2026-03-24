# ================================================================================================
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import joblib


# ------------------------------------DATA_CLEANING--------------------------
def preprocess(data):
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
    data['Churn'] = data['Churn'].map({'Yes':1,'No':0})
    return data
import pandas as pd
data = pd.read_csv('new_data.csv')
data = preprocess(data)
# print(data.info())


target = data['Churn']
feature = data.drop('Churn',axis=1)

x_train,x_test,y_train,y_test = train_test_split(feature,target,test_size=0.2,random_state=42)

# smote = SMOTE(random_state=42)

# x_res,y_res = smote.fit_resample(x_train,y_train)
from sklearn.utils import resample

# Combine X_train and y_train
new = pd.concat([x_train, y_train], axis=1)

# Separate classes
majority = new[new.Churn == 0]
minority = new[new.Churn == 1]

# Undersample majority
majority_downsampled = resample(majority,
                                replace=False,
                                n_samples=len(minority),
                                random_state=42)

# Combine minority + downsampled majority
downsampled = pd.concat([majority_downsampled, minority])

# Shuffle
downsampled = downsampled.sample(frac=1, random_state=42)

# print(downsampled.Churn.value_counts())


x_res = downsampled.drop('Churn',axis=1)
y_res = downsampled['Churn']
print(y_res.value_counts())


from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

model1 = XGBClassifier()
model2 = RandomForestClassifier(n_estimators=100,random_state=42)
models = [model1,model2]

# y1 = model1.predict(x_test)
# y2 = model2.predict(x_test)
# print(confusion_matrix(y_test,y1))
# print(confusion_matrix(y_test,y2))
# data = []
# for model in models:
#     temp = cross_val_score(model,x_res,y_res,cv=5,scoring='accuracy')
#     data.append(temp)
# print(data)

models_store = {}

for model in models:
    model.fit(x_res,y_res)
    models_store[model.__class__.__name__] = model

    # y = model.predict(x_test)
    # print('CONFUSION MATRIX-> \n',confusion_matrix(y_test,y))
    # print('CLASSIFICATION REPORT-> \n',classification_report(y_test,y))

joblib.dump({
    'models':models_store,
    'preprocessor':preprocess
},'pipeline.pkl')


# models = [model1,model2]
# data = []
# for i in models:
#     i.fit(x_res,y_res)
#     print(i.class)