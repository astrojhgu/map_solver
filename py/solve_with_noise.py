#!/usr/bin/env python
import argparse
import scipy.io
from scipy.sparse.linalg import lgmres, gmres, LinearOperator
import scipy
import numpy as np
from numpy.fft import rfft, irfft
import sys

def deconv(signal, kernel):
    s=rfft(signal)
    k=rfft(kernel)
    s[1:]/=k[1:]
    #s/=k
    return irfft(s)


parser = argparse.ArgumentParser(description='solve brust')
parser.add_argument('--tod', nargs=1, required=True)
parser.add_argument('--pmat', nargs=1, required=True)
parser.add_argument('--noise', nargs=1, required=True)
args=parser.parse_args()
#print(args.pmat)


scan_matrix=scipy.io.mmread(args.pmat[0])
tod=scipy.io.mmread(args.tod[0]).squeeze()
noise=scipy.io.mmread(args.noise[0]).squeeze()
#answer=scipy.io.mmread(sys.argv[4]).squeeze()


cnt=0

def A_func(x):
    global cnt
    cnt+=1
    return scan_matrix.T*deconv(scan_matrix*x, noise)


def A_func_naive(x):
    global cnt
    cnt+=1
    return scan_matrix.T*scan_matrix*x


b=scan_matrix.T*deconv(tod, noise)
b_naive=scan_matrix.T*tod

A=LinearOperator((len(b), len(b)), matvec=A_func)
A_naive=LinearOperator((len(b), len(b)), matvec=A_func_naive)

def cb(x):
    global cnt
    cnt+=1
    if cnt%100==0:
        print(cnt,np.sum(x**2))

tol=1e-16
answer, result=gmres(A_naive, b_naive, tol=tol, atol=tol, maxiter=100000, callback=cb)

expected_y=scan_matrix*answer;

resid=expected_y-scan_matrix*answer
print(np.sum(resid**2))

print(np.sum((A_naive(answer)-scan_matrix.T*expected_y)**2))
resid1=A(answer)-scan_matrix.T*deconv(expected_y, noise)
print(np.sum(resid1**2))
sys.exit()

solution, result=gmres(A, b, x0=answer, tol=tol, atol=tol, maxiter=100000, callback=cb)


#solution, result=lgmres(ata, b, tol=1e-20, atol=1e-20)
print(solution)
scipy.io.mmwrite('brute', np.atleast_2d(solution).T)
