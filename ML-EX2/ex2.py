##Boaz Ardel - 203642806##
import os
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def generateExamples(Examples_num, Class_num, u, sig):
    return np.vstack([np.random.normal(u, sig, Examples_num), np.ones(Examples_num) * Class_num]).T


def softmax(w, x, b):
    """Compute softmax values for each sets of scores in x."""
    t = x * w + b   #np.dot
    return np.exp(t) / np.sum(np.exp(t), axis=0)

def gradientW(w,xt,b,yt):
    S = softmax(w,xt,b)
    Syt = S[int(yt)-1]*xt - xt
    S = S*xt
    S[int(yt)-1] = Syt
    return S

def gradientB(w,xt,b,yt):
    S = softmax(w,xt,b)
    S[int(yt)-1] = S[int(yt)-1] - 1
    return S

def normal(x,mu,sigma):
    return ( 2.*np.pi*sigma**2. )**-.5 * np.exp( -.5 * (x-mu)**2. / sigma**2. )

def main():
    classExample_1 = generateExamples(100, 1, 2, 1)
    classExample_2 = generateExamples(100, 2, 4, 1)
    classExample_3 = generateExamples(100, 3, 6, 1)

    ExampleSet = np.concatenate((classExample_1, classExample_2, classExample_3), axis=0)

    '''
    print "1:" + str(classExample_1)
    print "2:" + str(classExample_2)
    print "3:" + str(classExample_3)
    print ">>>>>>>>>>>>>>>>>>>>>>>\n" + str(np.random.permutation(ExampleSet))
    '''
    W = np.ones(3)
    b = np.ones(3) * 0.5
    eta = 0.01
    epochs = 50

    for epoch in range(1, epochs):
        for [x, y] in np.random.permutation(ExampleSet):  # shuffled example set
            #print str(y) + ">>>" + str(softmax(W, x, b)) #debug
            #Calc Loss(Yt,Softmax(Xt))
            W_old = W
            W = W - eta*gradientW(W,x,b,y)
            b = b - eta*gradientB(W_old,x,b,y)

    '''Printing:'''
    t = np.linspace(0, 10, 200) #0.05 sample
    vfunc = np.vectorize(normal)
    normdist1 = vfunc(t, 2, 1)
    normdist2 = vfunc(t, 4, 1)
    normdist3 = vfunc(t, 6, 1)
    a = normdist1 / (normdist1+normdist2+normdist3)

    softdist = []
    for i in t:
        softdist.append(softmax(W,i,b)[0])

    plt.interactive(False)
    plt.plot(t, a, 'r') # plotting t, a - normal dist
    plt.plot(t, softdist, 'b')  # plotting t, b - softmax prob
    red_patch = mpatches.Patch(color='red', label='Normal Distribution')
    blue_patch = mpatches.Patch(color='blue', label='SGD')
    plt.legend(handles=[red_patch,blue_patch])
    plt.title('Distribution Differences for ' + str(epochs) + ' epochs')
    plt.ylabel('Probability')
    plt.xlabel('x')
    plt.show(block=True)
if __name__ == "__main__":
    main()