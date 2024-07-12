
#!/usr/bin/env python3
#cython: language_level=3
# -*- coding: utf-8 -*-
# distutils: language = c++



import time

import random
import concurrent.futures
import concurrent.futures

from ._utils cimport *

import numpy
cimport numpy
cimport cython
numpy.import_array()

from cython.parallel cimport prange

from libcpp.map cimport map as mapp

from libc.stdlib cimport calloc, free
from libc.stdio cimport FILE, fopen, fwrite, fscanf, fclose, fprintf
# cython: profile=True


"""
Wraper for C EMD implementation
"""


cdef public list SUITS = ['C', 'D', 'H', 'S']
cdef public list VALUES = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']

cdef public dict SUITS_INDEX = {'C': 0, 'D': 1, 'H': 2, 'S': 3}
cdef public dict VALUES_INDEX = {'2': 0, '3': 1, '4': 2, '5': 3, '6': 4, '7': 5, '8': 6, '9': 7, 'T': 8, 'J': 9, 'Q': 10, 'K': 11, 'A': 12}


ctypedef numpy.uint8_t uint8
ctypedef numpy.uint16_t uint16
ctypedef numpy.int16_t int16
ctypedef numpy.int32_t int32
ctypedef numpy.float32_t float32
ctypedef numpy.float64_t float64
ctypedef numpy.int64_t int64

ctypedef numpy.npy_bool boolean



#####################
#####################


######################
# https://github.com/obachem/kmc2/blob/master/kmc2.pyx
# 
###################### 
def kmc2(float32[:, :] X, k, chain_length=200, afkmc2=True, random_state=None, weights=None): 
    """Cython implementation of k-MC2 and AFK-MC2 seeding
    
    Args:
      X: (n,d)-shaped numpy.ndarray with data points (or scipy CSR matrix)
      k: number of cluster centers
      chain_length: length of the MCMC chain
      afkmc2: Whether to run AFK-MC2 (if True) or vanilla K-MC2 (if False)
      random_state: numpy.random.RandomState instance or integer to be used as seed
      weights: n-sized numpy.ndarray with weights of data points (default: uniform weights)
    Returns:
      (k, d)-shaped numpy.ndarray with cluster centers
    """
    # Local cython variables
    cdef numpy.intp_t j, curr_ind
    cdef double cand_prob, curr_prob
    cdef double[::1] q_cand, p_cand, rand_a
    cdef int N_CARDS = 24

    #X = numpy.fromiter(X, numpy.float32[:], X.shape[0])

    cdef float32[:] ALL_X = numpy.zeros(X.shape[0], dtype = numpy.float32)

    cdef float32[:] all_values = numpy.zeros(X.shape[1]*2, dtype = numpy.float32)
    cdef float32[:] all_valuesa = numpy.zeros(X.shape[1]*2 - 1 , dtype = numpy.float32)
    cdef float32[:] all_valuesaa = numpy.zeros(X.shape[1]*2 - 1, dtype = numpy.float32)
    cdef float32[:] deltaaa = numpy.zeros(X.shape[1]*2 - 1, dtype = numpy.float32)

    cdef float32 [:] ALL_X_VIEW = ALL_X[:]

    print('copying provided X to force memmap dereference. (experience segfault when calculating centers during the same program when the prob_dist memmap file is made.)')
    cdef numpy.ndarray[float32, ndim=2] XX = numpy.zeros((X.shape[0], X.shape[1]), dtype = numpy.float32)
    numpy.copyto(XX, X)


    sparse = False
    if weights is None:
        weights = numpy.ones(X.shape[0], dtype=numpy.double)
    if random_state is None or isinstance(random_state, int):
        random_state = numpy.random.RandomState(random_state)
    if not isinstance(random_state, numpy.random.RandomState):
        raise ValueError("RandomState should either be a numpy.random.RandomState"
                         " instance, None or an integer to be used as seed.")


    cntrs = numpy.zeros((k, X.shape[1]), dtype=numpy.float32)

    #cntrs[0] = X[random_state.choice(X.shape[0], p=weights/weights.sum())]

    cdef float32[:,:] ctr = cntrs[:]
    
    #cdef float32[:] ct = cntrs[0]
    
    print(X.shape)
    print(XX[0])
    print(emd_c(1, XX[0], ctr[0], all_values, all_valuesa, all_valuesaa, deltaaa))

    if afkmc2:
        
        # assumption free proposal distribution using EMD_C 
        for j in range(ALL_X_VIEW.shape[0]):
            ALL_X_VIEW[j] = emd_c(1, XX[j], ctr[0], all_values, all_valuesa, all_valuesaa, deltaaa)
        # following q value pulled from adriangoe afkmc2

        q = numpy.fromiter(ALL_X_VIEW, numpy.double, count=X.shape[0])

        print(q)
        q = q/(2*numpy.sum(q)) + 1/(2 * X.shape[0])

        #di = numpy.min(numpy.fromiter(ALL_X_VIEW, numpy.double, count=X.shape[0])) 
        #print(di)
        
        #q = di/numpy.sum(di, dtype = numpy.double) + weights/numpy.sum(weights)  # Only the potentials
        #print(q)
        
    else:
        q = numpy.copy(weights)
    # Renormalize the proposal distribution
    q = q / numpy.sum(q)
    print(q)
    for i in range(k-1):
        t1 = time.time()
        # Draw the candidate indices
        cand_ind = random_state.choice(X.shape[0], size=(chain_length), p=q).astype(dtype=numpy.intp)
        
        # Extract the proposal probabilities
        q_cand = q[cand_ind].astype(numpy.double)
    
        # shortest distance from previous centers using EMD_C 
        dist = [[emd_c(1, xi, XX[cand_ind[yi]], all_values, all_valuesa, all_valuesaa, deltaaa) for yi in range(len(cand_ind))] for xi in ctr[0:(i+1)]]
        
        # Compute potentials
    
        p_cand = numpy.min(dist).astype(numpy.double)*weights[cand_ind]
        
        # Compute acceptance probabilities
        rand_a = random_state.random_sample(size=(chain_length))
        with cython.boundscheck(False), cython.wraparound(False), cython.cdivision(True):
            # Markov chain
            for j in range(q_cand.shape[0]):
                cand_prob = (p_cand[j])/(q_cand[j])
                if j == 0 or curr_prob == 0.0 or cand_prob/curr_prob > rand_a[j]:
                    # Init new chain             Metropolis-Hastings step
                    curr_ind = j
                    curr_prob = cand_prob
        # centers[i+1, :] = rel_row.todense().flatten() if sparse else rel_row
        ctr[i+1] = X[cand_ind[curr_ind]]
        t2=time.time()
        t = t2-t1
        print('~' + str(t*(k - 1 - i)//60) + ' minutes until finished. ' + str(100*(i/(k-1)))[:4] + '% done     ', end = '\r')
    return cntrs



#################################################################################################
# My code below.
#################################################################################################
#
# Taken from stack overflow. Fast list combination generator for C(n, k)
#

def nump2(n, k):
    a = numpy.ones((k, n-k+1), dtype=numpy.int16)
    a[0] = numpy.arange(n-k+1)
    for j in range(1, k):
        reps = (n-k+j) - a[j-1]
        a = numpy.repeat(a, reps, axis=1)
        ind = numpy.add.accumulate(reps)
        a[j, ind[:-1]] = 1-reps[1:]
        a[j, 0] = j
        a[j] = numpy.add.accumulate(a[j])
    return a.T

#
# Taken from stack overflow. Fast list combination generator for C(n, k)
#

# 
# See if a card appears twice in the hand/table combo. Can be faster.
# 


@cython.boundscheck(False) 
@cython.wraparound(False)
cdef numpy.npy_bool contains_duplicates(int16[:] XX) nogil:
    cdef unsigned int count
    cdef unsigned int length
    cdef unsigned int countt = 0
    length = 7
    for count in range(length-1):        
        for countt in range(count+1, length):
            if(XX[count] == XX[countt]):
                return True
        
    return False


# 
# See if a card appears twice in the hand/table combo. Can be faster.
# 


@cython.boundscheck(False) 
@cython.wraparound(False)
cdef numpy.npy_bool contains_duplicates_turn(int16[:] XX) nogil:
    cdef unsigned int count
    cdef unsigned int length
    cdef unsigned int countt = 0
    length = 6
    for count in range(length-1):        
        for countt in range(count+1, length):
            if(XX[count] == XX[countt]):
                return True
        
    return False

@cython.boundscheck(False) 
@cython.wraparound(False)
cdef numpy.npy_bool contains_duplicates_flop(int16[:] XX) nogil:
    cdef unsigned int count
    cdef unsigned int length
    cdef unsigned int countt = 0
    length = 5
    for count in range(length-1):        
        for countt in range(count+1, length):
            if(XX[count] == XX[countt]):
                return True
        
    return False

# 
# Check if the hand/board combo (XX) contains a given card (comp)
# 


@cython.boundscheck(False)
@cython.wraparound(False)
cdef numpy.npy_bool contains(int16[:] XX, int comp) nogil:
    cdef unsigned int count
    cdef unsigned int length = 7
    for count in range(length):
        if(XX[count] == comp):
            return True
        
    return False


@cython.boundscheck(False)
@cython.wraparound(False)
cdef numpy.npy_bool contains_flop(int16[:] XX, int comp) nogil:
    cdef unsigned int count
    cdef unsigned int length = 5
    for count in range(length):
        if(XX[count] == comp):
            return True
        
    return False

@cython.boundscheck(False)
@cython.wraparound(False)
cdef numpy.npy_bool contains_turn(int16[:] XX, int comp) nogil:
    cdef unsigned int count
    cdef unsigned int length = 6
    for count in range(length):
        if(XX[count] == comp):
            return True
        
    return False

# 
# Get the Expected Hand Strength for a given hand/board river combination.
# Returns a 'tally' array for the total number of ties, wins, and losses.
# This is an exact calculation so I leave the result as an integer ratio 
# in order to avoid precision loss later down the line.
#  




@cython.boundscheck(False) 
@cython.wraparound(False)
cdef void get_ehs_fast(int16[:] j, int16[:] twl_tiewinloss) nogil:
    
    cdef int T_CARDS = 5
    cdef int N_CARDS = 24

    cdef int16 x[45]
    cdef int16 i, k
    cdef int hero, v, c
    cdef unsigned long long mask = 0
    cdef unsigned int seven = 7
    cdef unsigned long long one = 1
    cdef unsigned long long key

    
    c = 0
    for i in range(N_CARDS):
        if not contains(j, i):
            x[c] = i
            c += 1    

    mask |= one << j[2]
    mask |= one << j[3]
    mask |= one << j[4]
    mask |= one << j[5]
    mask |= one << j[6]

    hero = cy_evaluate(one << j[0] | one << j[1] | mask, seven)

    for i in range(0, N_CARDS - T_CARDS - 3):
        for k in range(i+1, N_CARDS - T_CARDS - 2):

            v = cy_evaluate(one << x[i] | one << x[k] | mask, seven)

            if(hero > v):
                twl_tiewinloss[1] += 1
            elif(v > hero):
                twl_tiewinloss[2] += 1
            else:
                twl_tiewinloss[0] += 1
    

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef int n_less(int16[:] X, int16 n) nogil:
    cdef: 
        int i, j
        int length_X = X.shape[0]
    i = 0
    for j in range(length_X):
        if (X[j] < n):
            i+=1
    return i





@cython.boundscheck(False)
cdef float32 emd_c(int p, float32[:] u_values, float32[:] v_values, float32[:] all_values,float32[:] all_valuesa, float32[:] all_valuesaa, float32[:] deltas, u_weights=None, v_weights=None) nogil:
    
    # the extra arrays after v_values are empty arrays to be filled by the funciton.
    # for best performance, it is up to you to define these empty arrays outside of heavy looping.
 

    insertion_sort_inplace_cython_float32(u_values)
    insertion_sort_inplace_cython_float32(v_values)

    
    cdef int i
    cdef int all_len = len(u_values) + len(v_values)
    
    
    for i in range(len(u_values)):
        all_values[i] = u_values[i]
    
    for i in range(len(u_values), all_len):
        all_values[i] = v_values[i-len(u_values)]

    insertion_sort_inplace_cython_float32(all_values)

    for i in range(all_len-1):
        deltas[i] = all_values[i+1] - all_values[i]
        
    


    return all_ind(u_values, v_values, all_values, all_valuesa, all_valuesaa, deltas)


@cython.boundscheck(False)
cpdef float32 emd_cp(int p, float32[:] u_values, float32[:] v_values, float32[:] _all_values,float32[:] _all_valuesa, float32[:] _all_valuesaa, float32[:] _deltas, u_weights=None, v_weights=None):
    r"""
    Compute, between two one-dimensional distributions :math:`u` and
    :math:`v`, whose respective CDFs are :math:`U` and :math:`V`, the
    statistical distance that is defined as:

    .. math::

        l_p(u, v) = \left( \int_{-\infty}^{+\infty} |U-V|^p \right)^{1/p}

    p is a positive parameter; p = 1 gives the Wasserstein distance, p = 2
    gives the energy distance.

    Parameters
    ----------
    u_values, v_values : array_like
        Values observed in the (empirical) distribution.
    u_weights, v_weights : array_like, optional
        Weight for each value. If unspecified, each value is assigned the same
        weight.
        `u_weights` (resp. `v_weights`) must have the same length as
        `u_values` (resp. `v_values`). If the weight sum differs from 1, it
        must still be positive and finite so that the weights can be normalized
        to sum to 1.

    Returns
    -------
    distance : float
        The computed distance between the distributions.

    Notes
    -----
    The input distributions can be empirical, therefore coming from samples
    whose values are effectively inputs of the function, or they can be seen as
    generalized functions, in which case they are weighted sums of Dirac delta
    functions located at the specified values.

    References
    ----------
    .. [1] Bellemare, Danihelka, Dabney, Mohamed, Lakshminarayanan, Hoyer,
           Munos "The Cramer Distance as a Solution to Biased Wasserstein
           Gradients" (2017). :arXiv:`1705.10743`.

    """
        



    return emd_c(1, u_values, v_values, _all_values, _all_valuesa, _all_valuesaa, _deltas)


def _validate_distribution(values, weights):
    """
    Validate the values and weights from a distribution input of `cdf_distance`
    and return them as ndarray objects.

    Parameters
    ----------
    values : array_like
        Values observed in the (empirical) distribution.
    weights : array_like
        Weight for each value.

    Returns
    -------
    values : ndarray
        Values as ndarray.
    weights : ndarray
        Weights as ndarray.

    """
    # Validate the value array.
    values = numpy.asarray(values, dtype=float)
    if len(values) == 0:
        raise ValueError("Distribution can't be empty.")

    # Validate the weight array, if specified.
    if weights is not None:
        weights = numpy.asarray(weights, dtype=float)
        if len(weights) != len(values):
            raise ValueError('Value and weight array-likes for the same '
                             'empirical distribution must be of the same size.')
        if numpy.any(weights < 0):
            raise ValueError('All weights must be non-negative.')
        if not 0 < numpy.sum(weights) < numpy.inf:
            raise ValueError('Weight array-like sum must be positive and '
                             'finite. Set as None for an equal distribution of '
                             'weight.')

        return values, weights

    return values, None

cpdef emd(int p, u_values, v_values, u_weights=None, v_weights=None):
    r"""
    Compute, between two one-dimensional distributions :math:`u` and
    :math:`v`, whose respective CDFs are :math:`U` and :math:`V`, the
    statistical distance that is defined as:

    .. math::

        l_p(u, v) = \left( \int_{-\infty}^{+\infty} |U-V|^p \right)^{1/p}

    p is a positive parameter; p = 1 gives the Wasserstein distance, p = 2
    gives the energy distance.

    Parameters
    ----------
    u_values, v_values : array_like
        Values observed in the (empirical) distribution.
    u_weights, v_weights : array_like, optional
        Weight for each value. If unspecified, each value is assigned the same
        weight.
        `u_weights` (resp. `v_weights`) must have the same length as
        `u_values` (resp. `v_values`). If the weight sum differs from 1, it
        must still be positive and finite so that the weights can be normalized
        to sum to 1.

    Returns
    -------
    distance : float
        The computed distance between the distributions.

    Notes
    -----
    The input distributions can be empirical, therefore coming from samples
    whose values are effectively inputs of the function, or they can be seen as
    generalized functions, in which case they are weighted sums of Dirac delta
    functions located at the specified values.

    References
    ----------
    .. [1] Bellemare, Danihelka, Dabney, Mohamed, Lakshminarayanan, Hoyer,
           Munos "The Cramer Distance as a Solution to Biased Wasserstein
           Gradients" (2017). :arXiv:`1705.10743`.

    """
    u_values, u_weights = _validate_distribution(u_values, u_weights)
    v_values, v_weights = _validate_distribution(v_values, v_weights)

    

    u_sorter = numpy.argsort(u_values)
    v_sorter = numpy.argsort(v_values)

    all_values = numpy.concatenate((u_values, v_values))
    all_values.sort(kind='mergesort')

    # Compute the differences between pairs of successive values of u and v.
    deltas = numpy.diff(all_values)

    # Get the respective positions of the values of u and v among the values of
    # both distributions.
    u_cdf_indices = u_values[u_sorter].searchsorted(all_values[:-1], 'right')
    v_cdf_indices = v_values[v_sorter].searchsorted(all_values[:-1], 'right')

    # Calculate the CDFs of u and v using their weights, if specified.
    if u_weights is None:
        u_cdf = u_cdf_indices / u_values.size
    else:
        u_sorted_cumweights = numpy.concatenate(([0],
                                              numpy.cumsum(u_weights[u_sorter])))
        u_cdf = u_sorted_cumweights[u_cdf_indices] / u_sorted_cumweights[-1]

    if v_weights is None:
        v_cdf = v_cdf_indices / v_values.size
    else:
        v_sorted_cumweights = numpy.concatenate(([0],
                                              numpy.cumsum(v_weights[v_sorter])))
        v_cdf = v_sorted_cumweights[v_cdf_indices] / v_sorted_cumweights[-1]
    print(u_cdf)
    print(v_cdf)
    # Compute the value of the integral based on the CDFs.
    # If p = 1 or p = 2, we avoid using numpy.power, which introduces an overhead
    # of about 15%.
    return numpy.sum(numpy.multiply(numpy.abs(u_cdf - v_cdf), deltas))
    






@cython.boundscheck(False) 
@cython.wraparound(False)
def do_calc(numpy.int64_t os, int16[:, :] x, int16[:, :] y, int dupes):
    cdef unsigned long long int total = 0
    cdef numpy.int64_t i, j, c
    cdef double t1, t2, t
    cdef numpy.int64_t x_shape = x.shape[0]
    cdef numpy.int64_t y_shape = y.shape[0]

    cd = 0

    # cdef numpy.ndarray[int16, ndim=2] z = numpy.empty((x_shape * y_shape, x.shape[1] + y.shape[1] + 3), dtype=numpy.int16)
    z_memmap = numpy.memmap('../results/river.npy', mode = 'r+', dtype = numpy.int16, shape = (x_shape * (y_shape - dupes), 3), offset = os )
    z_f_memmap = numpy.memmap('../results/prob_dist_RIVER.npy', mode = 'r+', dtype = numpy.float32, shape = (x_shape * (y_shape - dupes), 1), offset = os//3 * 2)
    #mp_memmap = numpy.memmap('../results/map.npy', mode = 'r+', dtype = numpy.ulonglong, shape = (x_shape * (y_shape - dupes), 1), offset = os//3 * 2)
    
    cdef int16 [:, :] z_view = z_memmap
    cdef numpy.float32_t [:, :] z_f_view = z_f_memmap
    
    #cdef unsigned long long [:, :] mp_view = mp_memmap

    cdef numpy.ndarray[int16, ndim=1] oh = numpy.empty(7, dtype=numpy.int16)
    cdef int16[:] oh_view = oh[:]

    cdef unsigned long long one = 1
    cdef unsigned long long key
    for i in range(x_shape):
        t1=time.time()
        for j in range(y.shape[0]):
            oh_view[:2] = x[i]
            oh_view[2:] = y[j]
            if(not contains_duplicates(oh_view)):
                
                get_ehs_fast(oh_view, z_view[cd])

                key = (one << oh_view[0]) | (one << oh_view[1]) | (one << oh_view[2]) | (one << oh_view[3]) | (one << oh_view[4]) | (one << oh_view[5]) | (one << oh_view[6]) 
                
                
                z_f_view[cd] = (z_view[cd][1]+.5*z_view[cd][0]) / (z_view[cd][1]+z_view[cd][0]+z_view[cd][2])

                #mp_view[cd] = key

                cd += 1
            
        t2=time.time()
        t = t2-t1
        print('~' + str(t*(x_shape - i)//60) + ' minutes until finished. ' + str(100*(i/x_shape))[:4] + '% done     ', end = '\r')
    # mp_memmap.flush()
    z_f_memmap.flush()
    z_memmap.flush()


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef void insertion_sort_inplace_cython_float32(float32[:] A) nogil:
    cdef: 
        int i, j
        numpy.float32_t key
        int length_A = A.shape[0]

    for j in range(1, length_A):
        key = A[j]
        i = j - 1
        while (i >= 0) & (A[i] > key):
            A[i + 1] = A[i]
            i = i - 1
        A[i + 1] = key


#set elents in A to the index they appear in B
@cython.boundscheck(False)
@cython.initializedcheck(False)
cdef float32 all_ind(float32[:] A, float32[:] AA, float32[:] B, float32[:] BB, float32[:] BBB, float32[:] deltas) nogil:
    cdef: 
        int32 i, j
        numpy.float32_t key, keyy, pd
        int32 length_A = len(A)
        int32 length_AA = len(AA)
        int32 length_B = len(B)

    pd = 0

    for j in range(length_B - 1):
        key = B[j]
        i = 0
        while (i < length_A) & (A[i] <= key):
            i = i + 1
        BB[j] = i/(length_A)

        i = 0
        while (i < length_AA) & (AA[i] <= key):
            i = i + 1
        BBB[j] = (i)/(length_AA)

        
        if(BB[j]>BBB[j]):
            pd += (BB[j] - BBB[j]) * deltas[j]
        else:
            pd += (BBB[j] - BB[j]) * deltas[j]

    return pd
        


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef void insertion_sort_inplace_cython_int16(int16[:] A):
    cdef: 
        int i, j
        int16 key
        int length_A = A.shape[0]

    for j in range(1, length_A):
        key = A[j]
        i = j - 1
        while (i >= 0) & (A[i] > key):
            A[i + 1] = A[i]
            i = i - 1
        A[i + 1] = key




@cython.boundscheck(False)
@cython.initializedcheck(False)
cpdef prob_dist_fun(int16[:, :] x, int16[:, :] y, float32[:, :] dist, float32[:, :] cntrs, int[:] lbs, int64 dupes, boolean turn, int64 os): #single byte offset. multiply by #cols and #bytes in data type

    cdef int T_CARDS
    cdef int N_CARDS
    if turn:
        T_CARDS = 5
        N_CARDS = 24
    else:
        T_CARDS = 4
        N_CARDS = 24
        
    

    cdef numpy.ndarray[int16, ndim=1] oh = numpy.empty(T_CARDS+1, dtype=numpy.int16)
    cdef numpy.ndarray[int16, ndim=2] oh_2d = numpy.empty((x.shape[0] * (y.shape[0] - dupes), T_CARDS+1 + 1), dtype=numpy.int16)
    cdef numpy.ndarray[int16, ndim=1] oh_z = numpy.empty(T_CARDS+2, dtype=numpy.int16)
    cdef numpy.ndarray[int16, ndim=1] oh_z_tmp = numpy.empty(T_CARDS + 2, dtype=numpy.int16)

    cdef mapp[unsigned long long, int] mp

    cdef numpy.ndarray[float32, ndim=2] prob_dist

    cdef int16[:] oh_view = oh[:]
    cdef int16[:, :] oh_view_2d = oh_2d[:]
    cdef int16[:] oh_z_view = oh_z[:]
    cdef int16[:] oh_z_view_tmp = oh_z_tmp[:]
    
    cdef int cc, c, i, j, tt, k, cd
    cdef long t

    cdef unsigned long long one = 1
    cdef unsigned long long key = -1

    cdef unsigned long long [:, :] mp_turn_view
    cdef numpy.ndarray[float32, ndim=1] mp_z

    cdef int16[:, :] yy = nump2(N_CARDS, T_CARDS)
    if(turn):
        ndupes = num_dupes(x, yy)
    else:
        ndupes = num_dupes_turn(x, yy)


    # SET UP MAPPING TO R + 1 DISTRIBUTIONS
    cd, c=0, 0
    for i in range(x.shape[0]):
        for j in range(yy.shape[0]):
            oh_z_view[:2] = x[i][:]
            oh_z_view[2:] = yy[j][:]
            if(not ((turn and contains_duplicates(oh_z_view)) or (not turn and contains_duplicates_turn(oh_z_view)))):
                if(turn):
                    key = (one << oh_z_view[0]) | (one << oh_z_view[1]) | (one <<oh_z_view[2]) | (one <<oh_z_view[3]) | (one <<oh_z_view[4]) | (one <<oh_z_view[5]) | (one << oh_z_view[6])
                else:
                    key = (one << oh_z_view[0]) | (one << oh_z_view[1]) | (one <<oh_z_view[2]) | (one <<oh_z_view[3]) | (one <<oh_z_view[4]) | (one <<oh_z_view[5])
                mp[key] = cd
                cd+=1

    # SET 2D MAT FOR NOGIL PARALLELISM - NOT WORKING FOR REASON UNKONWN ATM (PARALLELISM)
    for i in range(x.shape[0]):
        for j in range(y.shape[0]):
            oh_view[:2] = x[i][:]
            oh_view[2:] = y[j][:]

            if(not ((turn and contains_duplicates_turn(oh_view)) or (not turn and contains_duplicates_flop(oh_view)))):
                oh_view_2d[c][:-1] = oh_view[:]
                c += 1
    

    mp_z = numpy.zeros(len(dist[0]), dtype = numpy.float32)
    
    cdef float32[:] mp_z_view = mp_z[:] 
    
    # for each possible remaining card, we will store the subsequent index of the river cluster to which it responds.
    if turn:
        prob_dist = numpy.memmap('../results/prob_dist_TURN.npy', mode = 'r+', dtype = numpy.float32, shape = (x.shape[0] * (y.shape[0] - dupes), N_CARDS - T_CARDS - 2))[:] # 9 col
    else:
        prob_dist = numpy.memmap('../results/prob_dist_FLOP.npy', mode = 'r+', dtype = numpy.float32, shape = (x.shape[0] * (y.shape[0] - dupes), N_CARDS - T_CARDS - 2))[:] # 10 col -- both will need to change according to proper histogram schematics..
    
    cdef float32[:, :] prob_dist_memview = prob_dist[:]
    cdef float32[:] prob_dist_sing_memview = numpy.empty(N_CARDS-T_CARDS-1, dtype = numpy.float32)[:]

    cdef float32[:] all_values = numpy.empty(cntrs.shape[1] + dist.shape[1], dtype = numpy.float32)
    cdef float32[:] all_valuesa = numpy.empty(cntrs.shape[1] - 1 + dist.shape[1], dtype = numpy.float32)
    cdef float32[:] all_valuesaa = numpy.empty(cntrs.shape[1] - 1 + dist.shape[1], dtype = numpy.float32)
    cdef float32[:] deltaaa = numpy.empty(cntrs.shape[1] - 1 + dist.shape[1], dtype = numpy.float32)





    ii = cython.declare(cython.int)
    nn = cython.declare(cython.int, (x.shape[0] * (y.shape[0] - dupes)))
    ss = cython.declare(cython.int, 0)



    for ii in prange(nn, nogil=True, num_threads=16):
        for k in range(N_CARDS):
            if(not ((turn and contains_turn(oh_view_2d[ii], k)) or (not turn and contains_flop(oh_view_2d[ii], k)))):
                ## for each datapoint (float value for winning), find the best center given 
                
                
                oh_view_2d[ii][T_CARDS + 1] = k


                if(turn):
                    key = (one << oh_view_2d[ii][0])  | (one << oh_view_2d[ii][1]) | (one << oh_view_2d[ii][2])  | (one << oh_view_2d[ii][3])  | (one << oh_view_2d[ii][4])  | (one << oh_view_2d[ii][5])  | (one << oh_view_2d[ii][6])
                else:
                    key = (one << oh_view_2d[ii][0])  | (one << oh_view_2d[ii][1]) | (one << oh_view_2d[ii][2])  | (one << oh_view_2d[ii][3])  | (one << oh_view_2d[ii][4]) | (one << oh_view_2d[ii][5])
                

                
                #reward distance from losing.. temp logic until i understand paper
                #prob_dist_memview[ii][k] = emd_c(1, mp_z_view, cntrs[lbs[mp[key]]], all_values, all_valuesa, all_valuesaa, deltaaa)
                #with gil:
                #    print(k - n_less(oh_view_2d[ii], k))

                prob_dist_memview[ii][k - n_less(oh_view_2d[ii], k)] = emd_c(1, mp_z_view, cntrs[lbs[mp[key]]], all_values, all_valuesa, all_valuesaa, deltaaa)

    ## will delete soon. NOGIL PARALLELISM ABOVE :)
    '''
    for i in range(x.shape[0]):
        # load current portion of dataset to memory using the offset. 
        # fluff_view[:, :] = numpy.memmap('../results/fluffy.npy', mode = 'c', dtype = numpy.float32, shape = (y_shape_river[0], 1), offset = i * y_shape_river[0] * 8)[:]
        t1=time.time()
        
        for j in range(y.shape[0]):
            oh_view[:2] = x[i][:]
            oh_view[2:] = y[j][:]
            if(not ((turn and contains_duplicates_turn(oh_view)) or (not turn and contains_duplicates_flop(oh_view)))):
                oh_z_view[:T_CARDS+1] = oh_view
                oh_z_view_tmp[:] = oh_z_view

                prob_dist_sing_memview[:] = prob_dist_memview[cd][:]
                
                c = 0
                # print(cd)
                for k in range(N_CARDS):
                    if(not ((turn and contains_turn(oh_view, k)) or (not turn and contains_flop(oh_view, k)))):
                        ## for each datapoint (float value for winning), find the best center given 
                        
                        
                        oh_z_view[T_CARDS + 1] = k

                        # print()
                        # print()

                        # must sort table cards before using the mapping.. 
                        #insertion_sort_inplace_cython_int16(oh_z_view[2:])
                        
                        # print([oh_z_view[xxx] for xxx in range(T_CARDS+2)])
                        # print([oh_z_view_tmp[xxx] for xxx in range(T_CARDS+2)])

                        if(turn):
                            key = (one << oh_z_view[0])  | (one << oh_z_view[1]) | (one << oh_z_view[2])  | (one << oh_z_view[3])  | (one << oh_z_view[4])  | (one << oh_z_view[5])  | (one << oh_z_view[6])
                        else:
                            key = (one << oh_z_view[0])  | (one << oh_z_view[1]) | (one << oh_z_view[2])  | (one << oh_z_view[3])  | (one << oh_z_view[4]) | (one << oh_z_view[5])
                        
                        # offset key returned by os. 

                        # prob_dist_memview[cd][c] = emd_c(1, dist[mp[key]], cntrs[lbs[mp[key]]])

                        
                        #reward distance from losing.. temp logic until i understand paper
                        prob_dist_sing_memview[c] = emd_c(1, mp_z_view, cntrs[lbs[mp[key]]], all_values, all_valuesa, all_valuesaa, deltaaa)

                        #print([dist[mp[key]][jjj] for jjj in range(len(dist[mp[key]]))])
                        
                        #print([cntrs[lbs[mp[key]]][jjj] for jjj in range(len(cntrs[lbs[mp[key]]]))])
                        
                        

                        #oh_z_view[:] = oh_z_view_tmp
                        c += 1
                
                insertion_sort_inplace_cython_float32(prob_dist_sing_memview)
                prob_dist_memview[cd] = prob_dist_sing_memview[:]
                #print([prob_dist_memview[cd][jjj] for jjj in range(len(prob_dist_memview[cd]))])
                cd += 1
        
        t2=time.time()
        t = t2-t1
        print('~' + str(t*(x.shape[0] - i)//60) + ' minutes until finished. ' + str(100*(i/x.shape[0]))[:4] + '% done     ', end = '\r')
    '''
    prob_dist.flush()
    return dupes


@cython.boundscheck(False) 
@cython.wraparound(False)
cpdef flop_ehs(n, k, threads, new_file=False):
    adjcntrs = numpy.load('../results/adjcntrs_TURN.npy', mmap_mode = 'c')
    lbls = numpy.load('../results/lbls_TURN.npy', mmap_mode = 'c')
    
    cdef numpy.ndarray[float32, ndim = 2] cntrs = adjcntrs
    cdef int[:] lbs = lbls

    cdef mapp[unsigned long long, int] mp
    cdef numpy.ndarray[int16, ndim=1] oh = numpy.empty(k+1, dtype=numpy.int16)
    cdef int16[:] oh_view = oh[:]


    cdef int16[:, :] x = nump2(n, 2)
    cdef int16[:, :] y = nump2(n, k-1)

    dupes = num_dupes_turn(x, y)



    cdef numpy.ndarray[float32, ndim=2] dist = numpy.memmap('../results/prob_dist_TURN.npy', mode = 'c', dtype = numpy.float32, shape = (x.shape[0] * (y.shape[0] - dupes), n - k - 2))
    
    cdef float32[:, :] dist_view = dist[:]



    y = nump2(n, k-2)
    dupes = num_dupes_flop(x, y)


    if(new_file):
        flop_dist = numpy.memmap('../results/prob_dist_FLOP.npy', mode = 'w+', dtype = numpy.float32, shape = (x.shape[0] * (y.shape[0] - dupes), n - k - 1))
        flop_dist.flush()

    prob_dist_fun(x, y, dist_view, cntrs, lbs, dupes,  False, 0 * (y.shape[0]-dupes))

    return dupes

@cython.boundscheck(False) 
@cython.wraparound(False)
cpdef turn_ehs(n, k, threads, new_file=False):
    adjcntrs = numpy.load('../results/adjcntrs.npy', mmap_mode = 'c')
    lbls = numpy.load('../results/lbls.npy', mmap_mode = 'c')

    cdef float32[:, :] cntrs = adjcntrs
    cdef int[:] lbs = lbls

    cdef int16[:, :] x = nump2(n, 2)
    cdef int16[:, :] y = nump2(n, k)

    dupes = num_dupes(x, y)
    
    cdef unsigned long long one = 1
    cdef unsigned long long keyy

    cdef numpy.ndarray[float32, ndim=2] dist = numpy.memmap('../results/prob_dist_RIVER.npy', mode = 'c', dtype = numpy.float32, shape = (x.shape[0] * (y.shape[0] - dupes), 1))

    cdef float32[:, :] dist_view = dist[:]

    
    y = nump2(n, k-1)
    dupes = num_dupes_turn(x, y)

    if(new_file):
        prob_dist = numpy.memmap('../results/prob_dist_TURN.npy', mode = 'w+', dtype = numpy.float32, shape = (x.shape[0] * (y.shape[0] - dupes), n - k - 2))
        
        prob_dist.flush()


    prob_dist_fun(x, y, dist_view, cntrs, lbs, dupes, True, 0 * (y.shape[0]-dupes))
    
    return dupes


@cython.boundscheck(False) 
@cython.wraparound(False)
cpdef num_dupes_flop_cp(int16[:, :] x, int16[:, :] y):
    cdef unsigned long long int total = 0
    cdef numpy.int64_t i, j, c
    cdef double t1, t2, t
    cdef numpy.int64_t x_shape = x.shape[0]
    cdef numpy.int64_t y_shape = y.shape[0]

    cdef mapp[unsigned long long, int] mp

    cd = 0

    cdef numpy.ndarray[int16, ndim=1] oh = numpy.empty(5, dtype=numpy.int16)
    cdef int16 [:] oh_view = oh

    
    for j in range(y_shape):
        oh[:2] = x[0][:]
        oh[2:] = y[j][:]
        if(contains_duplicates_flop(oh)):
            cd += 1
    return cd


@cython.boundscheck(False) 
@cython.wraparound(False)
cdef num_dupes_flop(int16[:, :] x, int16[:, :] y):
    cdef unsigned long long int total = 0
    cdef numpy.int64_t i, j, c
    cdef double t1, t2, t
    cdef numpy.int64_t x_shape = x.shape[0]
    cdef numpy.int64_t y_shape = y.shape[0]

    cdef mapp[unsigned long long, int] mp

    cd = 0

    cdef numpy.ndarray[int16, ndim=1] oh = numpy.empty(5, dtype=numpy.int16)
    cdef int16 [:] oh_view = oh

    
    for j in range(y_shape):
        oh[:2] = x[0][:]
        oh[2:] = y[j][:]
        if(contains_duplicates_flop(oh)):
            cd += 1
    return cd

@cython.boundscheck(False) 
@cython.wraparound(False)
cdef num_dupes_turn(int16[:, :] x, int16[:, :] y):
    cdef unsigned long long int total = 0
    cdef numpy.int64_t i, j, c
    cdef double t1, t2, t
    cdef numpy.int64_t x_shape = x.shape[0]
    cdef numpy.int64_t y_shape = y.shape[0]

    cdef mapp[unsigned long long, int] mp

    cd = 0

    cdef numpy.ndarray[int16, ndim=1] oh = numpy.empty(6, dtype=numpy.int16)
    cdef int16 [:] oh_view = oh

    
    for j in range(y_shape):
        oh[:2] = x[0]
        oh[2:] = y[j]
        if(contains_duplicates_turn(oh)):
            cd += 1
    return cd


@cython.boundscheck(False) 
@cython.wraparound(False)
cdef num_dupes(int16[:, :] x, int16[:, :] y):
    cdef unsigned long long int total = 0
    cdef numpy.int64_t i, j, c
    cdef double t1, t2, t
    cdef numpy.int64_t x_shape = x.shape[0]
    cdef numpy.int64_t y_shape = y.shape[0]

    cdef mapp[unsigned long long, int] mp

    cd = 0

    cdef numpy.ndarray[int16, ndim=1] oh = numpy.empty(7, dtype=numpy.int16)
    cdef int16 [:] oh_view = oh

    
    for j in range(y_shape):
        oh[:2] = x[0]
        oh[2:] = y[j]
        if(contains_duplicates(oh)):
            cd += 1
    return cd


@cython.boundscheck(False) 
@cython.wraparound(False)
def river_ehs(n, k, threads, new_file=False):
    x = nump2(n, 2)
    y = nump2(n, k)

    dupes = num_dupes(x, y)
    

    if(new_file):
        z = numpy.memmap('../results/river.npy', mode = 'w+', dtype = numpy.int16, shape = ((y.shape[0] - dupes) * x.shape[0], 3))
        z_f = numpy.memmap('../results/prob_dist_RIVER.npy', mode = 'w+', dtype = numpy.float32, shape = ((y.shape[0] - dupes) * x.shape[0], 1))
        #mp = numpy.memmap('../results/map.npy', mode = 'w+', dtype = numpy.ulonglong, shape = ((y.shape[0] - dupes) * x.shape[0], 1))
            
        z.flush()
        z_f.flush()
        #mp.flush()

    chunksize = len(x) // (threads-1)
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # change threads to appropriate number of workers for your system
        futures = []
        for i in range(threads-1):
            strt = i * chunksize
            stp = ((i + 1) * chunksize) if i != (threads - 2) else len(x)
            
            futures.append(executor.submit(do_calc, strt * (y.shape[0]-dupes) * 3 * 2, x[strt:stp], y, dupes))
        concurrent.futures.wait(futures)

        output = [f.result() for f in futures]
    
    return dupes



@cython.boundscheck(False) 
@cython.wraparound(False)
def test():
    print(handtype(card_str_to_int("AC") | card_str_to_int("AD") | card_str_to_int("AS") | card_str_to_int("5S") | card_str_to_int("JH") | card_str_to_int("JC") | card_str_to_int("JD"), 7))