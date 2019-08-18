##Boaz Ardel - 203642806##
import numpy as np
import pickle

relu = lambda x: np.maximum(x, 0)

def reluDerivative(x):
    x[x<=0] = 0
    x[x>0] = 1
    return x

def softmax(t):
    return np.exp(t) / np.sum(np.exp(t), axis=0)

def sigmoid(x, derivative=False):
    if (derivative == True):
        return sigmoid(x) * sigmoid(1 - x)
    return 1 / (1 + np.exp(-x))

def tanh(x,der=False):
    if(der == True):
        return (1-(np.tanh(x) ** 2))
    return np.tanh(x)

def Forward_Prop(x,y, params):
    W1, b1, W2, b2 = [params[key] for key in ('W1', 'b1', 'W2', 'b2')]
    z1 = np.dot(W1,x[np.newaxis].T) + b1        #x transposed because of input from dataset
    h1 = relu(z1)
    z2 = np.dot(W2, h1) + b2
    #z2 = z2 / np.linalg.norm(z2)
    y_hat = softmax(z2)

    loss = 1
    if(int(np.argmax(y_hat))==int(y)):
        loss = 0        #sum of the log-probability of elemnt Yt (index is y)

    return {'x': x, 'y': y, 'z1': z1, 'h1': h1, 'z2': z2, 'y_hat': y_hat, 'loss': loss}

def Back_Prop(results,params):
    W1, b1, W2, b2 = [params[key] for key in ('W1', 'b1', 'W2', 'b2')]
    x, y, z1, h1, z2, y_hat, loss = [results[key] for key in ('x', 'y', 'z1', 'h1', 'z2', 'y_hat', 'loss')]

    k = y_hat**2
    k[int(y)] = y_hat[int(y)] * (y_hat[int(y)] - 1)     #dL/dz2

    dW2 = np.dot(((int(y)/y_hat) * k) , h1.T)           #dL/dW2 = dL/dz2 * dz2/dW2 = y/y_hat * k * h
    db2 = (int(y)/y_hat) * k                            #dL/dW2 = dL/dz2 * dz2/db2 = y/y_hat * k

    #z1 = z1 / np.linalg.norm(z1)
    dW1_temp = np.dot(W2.T, db2) * reluDerivative(z1)   #dL/dW1 = dL/dz2 * dz2/dh * dh/dz1 * dz1/dW1 = y/y_hat * k * g'(z1) * x
    dW1 = np.dot(dW1_temp , x[np.newaxis])
    db1 = dW1_temp                                      #dL/dW1 = dL/dz2 * dz2/dh * dh/dz1 * dz1/db1 = y/y_hat * k * g'(z1)

    return {'db1': db1, 'dW1': dW1, 'db2': db2, 'dW2': dW2}

def update_Gradients(param_gradiants,params,eta):
    W1, b1, W2, b2 = [params[key] for key in ('W1', 'b1', 'W2', 'b2')]
    db1, dW1, db2, dW2 = [param_gradiants[key] for key in ('db1', 'dW1', 'db2', 'dW2')]

    W1 = W1 - eta * dW1
    W2 = W2 - eta * dW2
    b1 = b1 - eta * db1
    b2 = b2 - eta * db2

    return {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

def Unison_shuffle(x,y):
    seed = np.random.randint(0, 10)  # Returns 0-9
    np.random.seed(seed)
    np.random.shuffle(x)
    np.random.seed(seed)
    np.random.shuffle(y)

def main():
    '''Load Stored Data'''
    (data_x, data_y, test_x) = pickle.loads(open("Stored.dat", "r").read())

    '''compute dataset size'''
    train_size = int(data_y.size * 0.8)
    Unison_shuffle(data_x, data_y)

    #train_x = data_x[:train_size,:]
    #train_y = data_y[:train_size]
    train_x = data_x
    train_y = data_y
    #verify_x = data_x[-(data_y.size - train_size):,:]
    #verify_y = data_y[-(data_y.size - train_size):]     #taking last 20% elements

    # Initialize random parameters and inputs
    eta = 0.0005
    epochs = 24
    H = 128        #Hidden layer size
    Y_Size = 10

    W1 = (np.random.rand(H,np.shape(train_x)[1])-0.5)*0.01
    b1 = (np.random.rand(H,1)-0.5)*0.01
    W2 = (np.random.rand(Y_Size,H)-0.5)*0.01
    b2 = (np.random.rand(Y_Size,1)-0.5)*0.01

    #np.seterr(all='warn')

    params = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

    for epoch in range(1, epochs):
        loss = 0
        Unison_shuffle(train_x, train_y)  # shuffled example set
        for x,y in zip(train_x,train_y):

            x = x/255.0     #NORM input
            results = Forward_Prop(x, y, params)

            loss += results['loss']      #for debug and parameters tuning

            param_gradiants = Back_Prop(results,params)
            params = update_Gradients(param_gradiants,params,eta)
        print str(epoch) + ':' +str(loss)

    final = open("test_x.pred",'w')
    for x in test_x:
        results = Forward_Prop(x,0, params)
        final.write(str(int(np.argmax(results['y_hat'])))+'\n')
    final.close()

if __name__ == "__main__":
    main()