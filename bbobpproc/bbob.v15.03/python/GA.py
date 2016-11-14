#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from random import random
from operator import add
from functools import reduce


def individual(length, min, max):
    ran = np.random.randint(min,max)
    return [ran for x in range(length)]

individual(2, 0, 500)


def population(count, length, min, max):
    return [individual(length, min, max) for x in range(count)]

population(30, 2, 0, 500)


def fitness(individual, target):
    length = len(individual)
    sum = reduce(add, length, 0)
    fitnesss = abs(target-sum)
    return fitnesss


def grade(pop, target):
    summed = reduce(add, (fitness(x, target) for x in pop), 0)
    return summed / (len(pop) * 1.0)

def evolve (pop, target, retain=0.2, random_select=0.05, mutate=0.01):
    graded = [ (fitness(x, target), x) for x in pop]
    graded = [pop[1] for x in sorted(graded)]
    retain_length = int(len(graded))
    parents = graded[:retain_length]

    for individual in graded[retain_length:]:
        if random_select > random():
            parents.append(individual)

    for individual in parents:
        if mutate > random():
            pos_to_mutate = np.random.randint(0, len(individual))
            individual[pos_to_mutate] = np.random.randint(0, min(individual))

    parents_length = len(parents)
    desired_length = len(pop) - parents_length
    children = []
    while len(children) < desired_length:
        male = np.random.randint(parents_length-1, 0)
        female = np.random.randint(parents_length-1, 0)

        i=+1
        print("loc " + str(i))
        if male != female:
            male = parents[male]
            female = parents[female]
            half = len(male) / 2
            child = male[:half] + female[half:]
            children.append(child)

    parents.extend(children)
    return parents


def tournament (pop, k):
    best = None
    for i in k:
        individual = pop[random(1, 5)]
        if best == None or fitness(individual) > fitness(best):
            best == individual
    return best

