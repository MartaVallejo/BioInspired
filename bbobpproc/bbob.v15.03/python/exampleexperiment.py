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
import GA

argv = sys.argv[1:] # shortcut for input arguments

datapath = 'PUT_MY_BBOB_DATA_PATH' if len(argv) < 1 else argv[0]

#dimensions = (2,4) if len(argv) < 2 else eval(argv[1])

dimensions = (2, 3, 5, 10, 20, 40) if len(argv) < 2 else eval(argv[1])
function_ids = bbobbenchmarks.nfreeIDs if len(argv) < 3 else eval(argv[2])
# function_ids = bbobbenchmarks.noisyIDs if len(argv) < 3 else eval(argv[2])
#instances = range(1, 6) + range(41, 51) if len(argv) < 4 else eval(argv[3])
instances = [1,2]

opts = dict(algid='PUT ALGORITHM NAME',
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

def nonlin(x,deriv=False):
        if(deriv==True):
            return x*(1-x)
        return 1/(1+np.exp(-x))

def PURE_RANDOM_SEARCH(fun, x, maxfunevals, ftarget):

    """samples new points uniformly randomly in [-5,5]^dim and evaluates
    them on fun until maxfunevals or ftarget is reached, or until
    1e8 * dim function evaluations are conducted.

    """


    #np.random.seed(1)
    dim = len(x)

    #X = np.random.randint(2,size=(dim+1,2))
    #Y = np.random.randint(2,size = (2,1))
    #print (X)
    X = np.random.random_integers(-5,5,(dim,2))
    Y = np.random.random_integers(-5,5,(dim,2))

    syn0 = np.random.random_integers(-1,1,(dim,2))
    syn1 = np.random.random_integers(-1,1,(dim,2))
    #print('syn0000 ' + str(syn0))
    for i in xrange(10):
        L0 = X
        GA.grade(syn0,ftarget)
        L1 = nonlin(np.dot(L0,syn0.T))
        L2 = nonlin(np.dot(L1,syn1))

        L2_error = Y - L2

        if (i % 1000) == 0:
            print "Error: " + str(np.mean(np.abs(L2_error)))

        L2_delta = L2_error*nonlin(L2,deriv=True)*(1-nonlin(L2,deriv=True))

        L1_error = L2_delta.dot(syn1.T)

        L1_delta = L1_error*nonlin(L1,deriv=True)*(1-nonlin(L1,deriv=True))

        #print('L2 Error ' + str(L2_error))

        L1dot = L1.dot(L2_delta)
        syn1 = syn1 + L1dot
        L2dot = L2.T.dot(L1_delta)
        syn0 = syn0 + L2dot.T
        print("Syn0 "+str(syn0))
        print()
        print("Syn1 " +str(syn1))
    maxfunevals = min(1e8 * dim, maxfunevals)
    popsize = min(maxfunevals, 200)
    fbest = np.inf

    for _ in range(0, int(np.ceil(maxfunevals / popsize))):
        xpop = 10. * np.random.rand(popsize, dim) - 5.
        fvalues = fun(xpop)
        idx = np.argsort(fvalues)
        if fbest > fvalues[idx[0]]:
            fbest = fvalues[idx[0]]
            xbest = xpop[idx[0]]
        if fbest < ftarget:  # task achieved
            break

    return xbest

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
