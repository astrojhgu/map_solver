#!/usr/bin/env python
import numpy as np
from scipy.io import mmread,mmwrite
from scipy.linalg import inv
from scipy.sparse import block_diag
import sys

n=int(sys.argv[1])
ptr=mmread('ptr_matrix.mtx')
tod=mmread('tod.mtx')
answer=mmread('answer.mtx')

ptr_mch=block_diag([ptr]*n)
answer_mch=np.concatenate([answer]*n,axis=1)
tod_mch=np.concatenate([tod]*n,axis=1)

mmwrite('ptr_{}_ch.mtx'.format(n), ptr_mch)
mmwrite('answer_{}_ch.mtx'.format(n), answer_mch)
mmwrite('tod_{}_ch.mtx'.format(n), tod_mch)
