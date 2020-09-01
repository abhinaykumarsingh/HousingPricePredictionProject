#Import Libraries
import numpy as np
import pandas as pd
import pickle
 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
 
#load data
df = pd.read_csv("new_class.csv")
 
# Split data
X= df.drop('price', axis=1)
y= df['price']
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=10)
 
# feature scaling
sc = StandardScaler()
sc.fit(X_train)
X_train = sc.transform(X_train)

X_test = sc.transform(X_test)
 
 
###### Load Model
 
model = pickle.load(open('bangalore_house_price_prediction_rfr_model.pkl','rb'))
 
 
# it help to get predicted value of house  by providing features value 
def predict_house_price(bath,balcony,total_sqft_int,bhk,price_per_sqft,location):
 
    x =np.zeros(len(X.columns)) # create zero numpy array, len = 107 as input value for model
 
  # adding feature's value accorind to their column index
    x[0]=bath
    x[1]=balcony
    x[2]=total_sqft_int
    x[3]=bhk
    x[4]=price_per_sqft
 
    if 'location_'+location in X.columns:
        loc_index = np.where(X.columns=="location_"+location)[0][0]
        x[loc_index] =1
 
  # feature scaling
    x = sc.transform([x])[0] # give 2d np array for feature scaling and get 1d scaled np array
 
    return model.predict([x])[0] # return the predicted value by train XGBoost model