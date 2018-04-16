import numpy
import scipy
from multiprocessing import Pool


def f(x):
    return x * x

def simulation_caller():
    with Pool(5) as p:
        print(p.map(f, [1, 2, 3]))

if  __name__ == "__main__":
    simulation_caller()