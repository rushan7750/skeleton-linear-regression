#importing numpy for array work
import numpy as np

X = np.array([1, 2]) #X values
Y = np.array([2, 4]) #Y values

X = X.reshape(-1, 1) #reshaping X into 2 dimensional array

np.save('data/x', X) #Saving the data
np.save('data/y', Y) #Saving the data... Again

del X, Y #deleting variables at the end
exit(0) #the function speaks for itself (exiting the program) 