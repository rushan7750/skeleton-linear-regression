import numpy as np #import numpy for array work
import joblib #importing joblib for saving final model

inputval = np.array([3]) #the input value. More like a question but only in numbers.
inputval = inputval.reshape(-1, 1) #converting the array into a array that our model expects(two dimensional array)

model = joblib.load('model/model.h5') #loading the trained model from model/model.h5 
result = model.predict(inputval) #predicting the result
print(f"Prediction: {result[0]}") #And now... Printing it(please don't mind the dot(.) at the end)

del model, inputval, result #deleting the variables at the end
exit(0) #the function speaks for itself (exiting the program) 