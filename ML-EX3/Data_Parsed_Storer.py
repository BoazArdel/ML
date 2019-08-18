import numpy as np
import pickle

data_y = np.loadtxt("train_y")
print 'train_y loaded successfully!'
data_x = np.loadtxt("train_x")
print 'train_x loaded successfully!'
test_x = np.loadtxt("test_x")
print 'test_x loaded successfully!'

with open("Stored.dat", "w") as f:
    f.write(pickle.dumps((data_x, data_y, test_x)))
