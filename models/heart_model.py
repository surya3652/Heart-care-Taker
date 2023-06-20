import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
# Loading Models
heart_model = RandomForestClassifier()

# Loading Dataset
df = pd.read_csv("F:\Files\BTECH\Projects\MINI_PROJECT\Proj\heart.csv")

# Data Preprocessing
x = df.drop('target',axis = 1)
y = df['target']

#Training the model
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.50,random_state = 60)
heart_model.fit(x_train,y_train)

# Loading the Model
y_pred = heart_model.predict(x_test)

# Accuracy of the model
print(accuracy_score(y_test,y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test,y_pred)

# Fitting the model with desired parameters
heart_model.fit(df[["ca", "cp", "exang", "thal", "oldpeak", "thalach", "age"]],df.target)

# Saving the model
pickle.dump(heart_model, open('heart_model.pkl','wb'))


