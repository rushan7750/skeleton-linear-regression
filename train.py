from sklearn.linear_model import LinearRegression #importing sklearn for linear regression
import numpy as np #import numpy for array work
import joblib #importing joblib for saving final model

X = np.load('data/x.npy') #loading X values
Y = np.load('data/y.npy') #loading Y values

model = LinearRegression() #creating instance of LinearRegression 

model.fit(X, Y) #training the model
r_sq = model.score(X, Y) #how well does our model understand the relatonship between X and Y
print('coefficient of determination:', r_sq) #print it(1.0 is 100%, 0.99 is 99% and so on) 

joblib.dump(model, 'model/model.h5') #save the model 

del X, Y, r_sq, model #deleting the variables at the end
exit(0) #the function speaks for itself (exiting the program) 