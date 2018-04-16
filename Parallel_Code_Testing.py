from multiprocessing import Pool
import numpy

def f(x):
	a = numpy.zeros((5,5) +x*x
	return a
if __name__ == '__main__':
	p = Pool(5)
	print(p.map(f, [1, 2, 3]))
