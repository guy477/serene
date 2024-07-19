
#!/usr/bin/env python3
#cython: language_level=3
# -*- coding: utf-8 -*-
# distutils: language = c++

from .._utils cimport *
from ..eval.eval cimport *

import numpy
cimport numpy
cimport cython
from libcpp.map cimport map as mapp
numpy.import_array()

import time
import random
import concurrent.futures
import concurrent.futures
from multiprocessing import Manager, set_start_method
from tqdm import tqdm


# cython: profile=True


set_start_method('spawn', force=True)



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
cdef void get_ehs_fast(int16[:] j, int32[:] twl_tiewinloss) nogil:
    
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
def do_calc(int16[:, :] x, int16[:, :] y):
    cdef unsigned long long int total = 0
    cdef numpy.int64_t i, j, c
    cdef numpy.int64_t x_shape = x.shape[0]
    cdef numpy.int64_t y_shape = y.shape[0]

    cdef dict local_dict = {}

    cdef numpy.ndarray[int16, ndim=1] oh = numpy.empty(7, dtype=numpy.int16)
    cdef int16[:] oh_view = oh[:]

    cdef unsigned long long one = 1
    cdef unsigned long long hand_board_combo, private_hand
    for i in tqdm(range(x_shape)):
        for j in range(y.shape[0]):
            oh_view[:2] = x[i]
            oh_view[2:] = y[j]
            if(not contains_duplicates(oh_view)):
                
                # Write the EHS for the hand/board combo "oh_view". Write this to z_view[cd]
                # TODO: Change z_view[cd] to be a dictionary mapping from 
                private_hand = (one << oh_view[0]) | (one << oh_view[1])
                hand_board_combo = private_hand | (one << oh_view[2]) | (one << oh_view[3]) | (one << oh_view[4]) | (one << oh_view[5]) | (one << oh_view[6]) 
                hand_type = handtype(hand_board_combo, 7)

                key = (private_hand, hand_type)

                if key not in local_dict:
                    local_dict[key] = numpy.zeros(3, dtype = numpy.int32)

                
                
                get_ehs_fast(oh_view, local_dict[key][:])

    return local_dict
    # mp_memmap.flush()
    # z_f_memmap.flush()
    # z_memmap.flush()


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
@cython.wraparound(False)
cdef get_abstracted_river_ehs(int16[:] j, river_ehs_dict):
    cdef int16 x[46]
    cdef int16 i, k
    cdef int v, c
    cdef unsigned long long mask = 0
    cdef unsigned int six = 6
    cdef unsigned int seven = 7
    cdef unsigned long long one = 1
    cdef unsigned long long key

    
    c = 0
    for i in range(52):
        if not contains(j, i):
            x[c] = i
            c += 1    

    mask |= one << j[0]
    mask |= one << j[1]
    mask |= one << j[2]
    mask |= one << j[3]
    mask |= one << j[4]
    mask |= one << j[5]
    
    cdef int T, W, L
    T = 0
    W = 0
    L = 0

    for i in range(0, c):
        
        TWL = river_ehs_dict[(one << j[0] | one << j[1], handtype(one << x[i] | mask, 7))]
        T += TWL[0]
        W += TWL[1]
        L += TWL[2]

    total_outcomes = W + L + T
    if total_outcomes == 0:
        return 0  # No outcomes recorded yet
    win_probability = W / total_outcomes
    tie_probability = T / total_outcomes
    return win_probability + 0.5 * tie_probability


@cython.boundscheck(False) 
@cython.wraparound(False)
cdef get_abstracted_turn_ehs(int16[:] j, turn_ehs_dict):
    cdef int16 x[47]
    cdef int16 i, k
    cdef int v, c
    cdef unsigned long long mask = 0
    cdef unsigned int six = 6
    cdef unsigned int seven = 7
    cdef unsigned long long one = 1
    cdef unsigned long long key
    cdef float TWL

    
    c = 0
    for i in range(52):
        if not contains(j, i):
            x[c] = i
            c += 1    

    mask |= one << j[0]
    mask |= one << j[1]
    mask |= one << j[2]
    mask |= one << j[3]
    mask |= one << j[4]
    # mask |= one << j[5]
    

    for i in range(0, c):
        
        TWL += turn_ehs_dict[(abstract_hand(one << j[0], one << j[1]), handtype_partial(one << x[i] | mask, 6))]
        # T += TWL[0]
        # W += TWL[1]
        # L += TWL[2]

    return TWL / c
            



@cython.boundscheck(False)
@cython.initializedcheck(False)
def turn_ehs_calc(int16[:, :] x, int16[:, :] y, river_ehs_dict): #single byte offset. multiply by #cols and #bytes in data type
    turn_ehs_dict = {}

    cdef numpy.ndarray[int16, ndim=1] oh = numpy.empty(6, dtype=numpy.int16)
    cdef int16[:] oh_view = oh[:]


    cdef unsigned long long one = 1
    for i in tqdm(range(x.shape[0])):
        for j in range(y.shape[0]):
            oh_view[:2] = x[i]
            oh_view[2:] = y[j]
            if(not contains_duplicates_turn(oh_view)):
                # hero = handtype(one << oh_view[0] | one << oh_view[1] | one << oh_view[2] | one << oh_view[3] | one << oh_view[4] | one << oh_view[5], 6)
                # Write the EHS for the hand/board combo "oh_view". Write this to z_view[cd]
                hero = (abstract_hand(one << oh_view[0], one << oh_view[1]), handtype_partial(one << oh_view[0] | one << oh_view[1] | one << oh_view[2] | one << oh_view[3] | one << oh_view[4] | one << oh_view[5], 6))

                ### TODO: Change z_view[cd] to be a dictionary mapping from 
                if hero not in turn_ehs_dict:
                    turn_ehs_dict[hero] = get_abstracted_river_ehs(oh_view, river_ehs_dict)
                else:
                    turn_ehs_dict[hero] = (turn_ehs_dict[hero] + get_abstracted_river_ehs(oh_view, river_ehs_dict))/2
        
    return turn_ehs_dict
    # mp_memmap.flush(


@cython.boundscheck(False)
@cython.initializedcheck(False)
def flop_ehs_calc(int16[:, :] x, int16[:, :] y, turn_ehs_dict): #single byte offset. multiply by #cols and #bytes in data type
    flop_ehs_dict = {}

    cdef numpy.ndarray[int16, ndim=1] oh = numpy.empty(5, dtype=numpy.int16)
    cdef int16[:] oh_view = oh[:]


    cdef unsigned long long one = 1
    for i in tqdm(range(x.shape[0])):
        for j in range(y.shape[0]):
            oh_view[:2] = x[i]
            oh_view[2:] = y[j]
            if(not contains_duplicates_flop(oh_view)):
                # hero = handtype(one << oh_view[0] | one << oh_view[1] | one << oh_view[2] | one << oh_view[3] | one << oh_view[4] | one << oh_view[5], 6)
                # Write the EHS for the hand/board combo "oh_view". Write this to z_view[cd]
                hero = (abstract_hand(one << oh_view[0], one << oh_view[1]), handtype(one << oh_view[0] | one << oh_view[1] | one << oh_view[2] | one << oh_view[3] | one << oh_view[4], 5))
                # TODO: Change z_view[cd] to be a dictionary mapping from 
                if hero not in flop_ehs_dict:
                    flop_ehs_dict[hero] = get_abstracted_turn_ehs(oh_view, turn_ehs_dict)
                else:
                    # cumulative average
                    flop_ehs_dict[hero] = (flop_ehs_dict[hero] + get_abstracted_turn_ehs(oh_view, turn_ehs_dict))/2
        
    return flop_ehs_dict

@cython.boundscheck(False) 
@cython.wraparound(False)
cpdef flop_ehs(n, k, turn_ehs_dict, threads = 8):
    x = nump2(n, 2)
    y = nump2(n, k-2)
    
    results = {}
    print(x.shape)
    chunksize = len(x) // (threads)
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # change threads to appropriate number of workers for your system
        futures = []
        for i in range(threads):
            strt = i * chunksize
            stp = ((i + 1) * chunksize) if i != (threads - 1) else len(x)
            print(x[strt:stp].shape)
            
            futures.append(executor.submit(flop_ehs_calc, x[strt:stp], y, turn_ehs_dict))
        concurrent.futures.wait(futures)

        output = [f.result() for f in futures]
        for i in output:
            results.update(i)
    return results

@cython.boundscheck(False) 
@cython.wraparound(False)
cpdef turn_ehs(n, k, river_ehs_dict, threads = 8):
    x = nump2(n, 2)
    y = nump2(n, k-1)
    
    results = {}
    print(x.shape)
    chunksize = len(x) // (threads)
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # change threads to appropriate number of workers for your system
        futures = []
        for i in range(threads):
            strt = i * chunksize
            stp = ((i + 1) * chunksize) if i != (threads - 1) else len(x)
            print(x[strt:stp].shape)
            
            futures.append(executor.submit(turn_ehs_calc, x[strt:stp], y, river_ehs_dict))
        concurrent.futures.wait(futures)

        output = [f.result() for f in futures]
        for i in output:
            results.update(i)
    return results


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
def river_ehs(n, k, threads):
    x = nump2(n, 2)
    y = nump2(n, k)

    results = {}

    chunksize = len(x) // (threads)
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # change threads to appropriate number of workers for your system
        futures = []
        for i in range(threads):
            strt = i * chunksize
            stp = ((i + 1) * chunksize) if i != (threads - 1) else len(x)
            
            futures.append(executor.submit(do_calc, x[strt:stp], y))
        concurrent.futures.wait(futures)

        output = [f.result() for f in futures]
        for i in output:
            results.update(i)
    return results



@cython.boundscheck(False) 
@cython.wraparound(False)
def test():
    print(handtype(card_str_to_int("AC") | card_str_to_int("AD") | card_str_to_int("AS") | card_str_to_int("5S") | card_str_to_int("JH") | card_str_to_int("JC") | card_str_to_int("JD"), 7))