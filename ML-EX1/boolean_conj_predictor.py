##Boaz Ardel - 203642806##
import os
import sys
import numpy as np

def Logic_Compute(Hypo, example_arr):
    if Hypo[0] == 0:
        return 0                #First Hypothesis
    Yt =[]
    for i in range(0,len(example_arr)):
        if Hypo[i] == 1:
            Yt.append(example_arr[i])
        elif Hypo[i] == 2:
            Yt.append(1 - example_arr[i]) #logic 'not' the value
    return int(np.logical_and.reduce(Yt))

def UpdateHypo(Hypo,instance):
    count = 0
    for val in instance:
        if val==1:
            if Hypo[count] == 2:    #0-neg&noneg, 1-noneg , 2-neg, 3-nonexist
                Hypo[count] = 3
            elif Hypo[count] == 0:
                Hypo[count] = 1
        else:
            if Hypo[count] == 1:    #0-neg&noneg, 1-noneg , 2-neg, 3-nonexist
                Hypo[count] = 3
            elif Hypo[count] == 0:
                Hypo[count] = 2
        count = count + 1
    return Hypo

def HypoPrint(Hypo):
    count = 1
    result = ""
    for i in Hypo:
        if i==1:
            result = result + "x" + str(count) + ","
        elif i==2:
            result = result + "not(x" + str(count) + "),"
        count = count + 1
    result = result[:-1]
    Output = open('output.txt','w')
    Output.write(result)
    Output.close()

def main(argv):
    if len(sys.argv) != 2:
        print "Argument Error Please follow EX1 instructions"
        exit()
    training_examples = np.loadtxt(sys.argv[1])

    d = training_examples[0].size - 1
    Hypo = np.zeros(d)              #0-neg&noneg, 1-noneg , 2-neg, 3-nonexist

    X = np.delete(training_examples,d,1)
    Y = training_examples[:,d]
    #print X,Y

    count = 0
    for instance in X:
        if((Logic_Compute(Hypo,instance)== 0) and (Y[count]==1)):
            Hypo = UpdateHypo(Hypo,instance)
        count = count + 1
    HypoPrint(Hypo)

if __name__ == "__main__":
   main(sys.argv[1:])
