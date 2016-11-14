#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Runs an entire experiment for benchmarking PURE_RANDOM_SEARCH on a testbed.

CAPITALIZATION indicates code adaptations to be made.
This script as well as files bbobbenchmarks.py and fgeneric.py need to be
in the current working directory.

Under unix-like systems:
    nohup nice python exampleexperiment.py [data_path [dimensions [functions [instances]]]] > output.txt &

"""
import sys # in case we want to control what to run via command line args
import time
import numpy as np
import fgeneric
import bbobbenchmarks
import random

from copy import deepcopy
from operator import add


argv = sys.argv[1:] # shortcut for input arguments

datapath = 'Results' if len(argv) < 1 else argv[0]

#dimensions = (2,4) if len(argv) < 2 else eval(argv[1])
dimensions = (2, 3, 5, 10, 20, 40) if len(argv) < 2 else eval(argv[1])
function_ids = bbobbenchmarks.nfreeIDs if len(argv) < 3 else eval(argv[2])
# function_ids = bbobbenchmarks.noisyIDs if len(argv) < 3 else eval(argv[2])
instances = range(1, 6) + range(41, 51) if len(argv) < 4 else eval(argv[3])


opts = dict(algid='NN&GA',
            comments='PUT MORE DETAILED INFORMATION, PARAMETER SETTINGS ETC')
maxfunevals = '10 * dim' # 10*dim is a short test-experiment taking a few minutes
# INCREMENT maxfunevals SUCCESSIVELY to larger value(s)
minfunevals = 'dim + 2'  # PUT MINIMAL sensible number of EVALUATIONS before to restart
maxrestarts = 10000      # SET to zero if algorithm is entirely deterministic


def run_optimizer(fun, dim, maxfunevals, ftarget=-np.Inf):
    """start the optimizer, allowing for some preparation.
    This implementation is an empty template to be filled

    """
    # prepare
    x_start = 8. * np.random.rand(dim) - 4

    # call, REPLACE with optimizer to be tested
    PURE_RANDOM_SEARCH(fun, x_start, maxfunevals, ftarget)




def PURE_RANDOM_SEARCH(fun, x, maxfunevals, ftarget):

    """samples new points uniformly randomly in [-5,5]^dim and evaluates
    them on fun until maxfunevals or ftarget is reached, or until
    1e8 * dim function evaluations are conducted.

    """


    neurons = 40
    chromolen = dim * neurons
    populationSize = 10
    pop = np.random.rand(populationSize, chromolen)
    traininst = 100
    mutation_rate = 0.1
    generations = 5
    count = 10

    #Random input as input is not known
    X = 2* np.random.rand(traininst,dim)

    #expected Output from COCO
    Y = fun(X)

    #activation function linear
    def mySigmoid(X):
        return 1 / (1 + np.exp(-X))

    #Neural network FF
    def NN (chromo, input):

        #random first weights
        syn0 = np.random.randint(-1,1,size=(dim,neurons))
        syn1 = chromo[chromolen - neurons:chromolen]

        #L0 worked out using the input throught activation function
        L0 = mySigmoid(input)
        #hiddenlayer found using input multiplied by first weights
        L1 = mySigmoid(np.dot(L0,syn0))
        #L2 output layer is the output from NN so can compare to expected
        L2 = mySigmoid(np.dot(L1,syn1))

        #return actual output from NN
        return L2

    #not used tried to create function to do crossover
    def crossover(chromo1,chromo2 ):
        #finds random position
        pos = int(random.random() * populationSize)
        #returns back chomosomes with crossed values
        return chromo1[:pos] + chromo2[pos:], chromo2[:pos] + chromo1[pos:]

    #Single gene mutation
    def mutate(chromo, mutation_rate):
        #for every element in the chromosome
        for i in xrange(chromolen):
            #random position
            pos_to_mutate = np.random.randint(0, chromolen)
            if mutation_rate > np.random.rand():
                #mutate the chromosome by replacing the random position in the chormosome with a random number
                chromo[pos_to_mutate ] = np.random.rand()
        return chromo

    def tournament(chromo):
        best = None
        individual = chromo[random.randint(1, 5)]
        if best == None or fitness(individual) > fitness(best):
            best == individual
        return best

    #not used as fitness was worked out using the error for my GA
    def fitness(chromo, ftarget):
        length = len(chromo)
        sum = reduce(add, length, 0)
        fitnesss = abs(ftarget - sum)
        return fitnesss
    #best fitness so far
    bestfit=0
    #best chromsome now
    bestchromo = 0

    #average error
    average = 0

    #hill climb

    #for each generation
    for gen in xrange(generations):
        for counts in xrange(count):
                #for each individual in the population
                for individual in xrange(populationSize):
                    #the error = the expected output minus the actual output
                    error = (Y[gen] - (NN(pop[individual], X[gen])))
                    if bestfit > error:
                        #keep the fittest
                        bestfit = error
                        bestchromo = pop[individual]
                    #print(error)

        #keep the best chromo
        pop[0] = bestchromo
        for individuals in range(1,populationSize):
            #mutate the population apart from the best found
            pop[individuals] = mutate(pop[individuals],mutation_rate)
            #work out error after mutation to see if its getting better
            aftermutation = (Y[gen] - (NN(pop[individuals],X[gen])))
            #trainederror =  ftarget - aftermutation
            #print(trainederror)

            #pop[individuals] = crossover(pop[individuals],pop[individuals-1])
            #pop[individuals] = tournament(pop[individuals])

            #the average error after mutation
            average = np.mean(aftermutation)
        print('The Average Error = ' + str(average))

    return average

        #for item in range(1,populationSize)



t0 = time.time()
np.random.seed(int(t0))

f = fgeneric.LoggingFunction(datapath, **opts)
for dim in dimensions:  # small dimensions first, for CPU reasons
    for fun_id in function_ids:
        for iinstance in instances:
            f.setfun(*bbobbenchmarks.instantiate(fun_id, iinstance=iinstance))

            # independent restarts until maxfunevals or ftarget is reached
            for restarts in xrange(maxrestarts + 1):
                if restarts > 0:
                    f.restart('independent restart')  # additional info
                run_optimizer(f.evalfun, dim,  eval(maxfunevals) - f.evaluations,
                              f.ftarget)
                if (f.fbest < f.ftarget
                    or f.evaluations + eval(minfunevals) > eval(maxfunevals)):
                    break

            f.finalizerun()

            print('  f%d in %d-D, instance %d: FEs=%d with %d restarts, '
                  'fbest-ftarget=%.4e, elapsed time [h]: %.2f'
                  % (fun_id, dim, iinstance, f.evaluations, restarts,
                     f.fbest - f.ftarget, (time.time()-t0)/60./60.))

        print '      date and time: %s' % (time.asctime())
    print '---- dimension %d-D done ----' % dim