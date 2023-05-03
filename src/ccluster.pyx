#!python
#cython: language_level=3
import time
import numpy
import random
import concurrent.futures

cimport numpy
cimport cython

from cython.parallel import prange


from libc.stdio cimport FILE, fopen, fwrite, fscanf, fclose, fprintf
# cython: profile=True
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.utils import check_array

ctypedef numpy.uint8_t uint8
ctypedef numpy.uint16_t uint16
ctypedef numpy.int16_t int16
ctypedef numpy.int64_t int64
ctypedef numpy.npy_bool boolean


#################################################################################################
# The below code is taken from eval7 - https://pypi.org/project/eval7/
#################################################################################################


cdef extern from "arrays.h":
    unsigned short N_BITS_TABLE[8192]
    unsigned short STRAIGHT_TABLE[8192]
    unsigned int TOP_FIVE_CARDS_TABLE[8192]
    unsigned short TOP_CARD_TABLE[8192]



cdef int CLUB_OFFSET = 0
cdef int DIAMOND_OFFSET = 13
cdef int HEART_OFFSET = 26
cdef int SPADE_OFFSET = 39

cdef int HANDTYPE_SHIFT = 24 
cdef int TOP_CARD_SHIFT = 16 
cdef int SECOND_CARD_SHIFT = 12 
cdef int THIRD_CARD_SHIFT = 8 
cdef int CARD_WIDTH = 4 
cdef unsigned int TOP_CARD_MASK = 0x000F0000 
cdef unsigned int SECOND_CARD_MASK = 0x0000F000 
cdef unsigned int FIFTH_CARD_MASK = 0x0000000F 

cdef unsigned int HANDTYPE_VALUE_STRAIGHTFLUSH = ((<unsigned int>8) << HANDTYPE_SHIFT)
cdef unsigned int HANDTYPE_VALUE_FOUR_OF_A_KIND = ((<unsigned int>7) << HANDTYPE_SHIFT)
cdef unsigned int HANDTYPE_VALUE_FULLHOUSE = ((<unsigned int>6) << HANDTYPE_SHIFT)
cdef unsigned int HANDTYPE_VALUE_FLUSH = ((<unsigned int>5) << HANDTYPE_SHIFT)
cdef unsigned int HANDTYPE_VALUE_STRAIGHT = ((<unsigned int>4) << HANDTYPE_SHIFT)
cdef unsigned int HANDTYPE_VALUE_TRIPS = ((<unsigned int>3) << HANDTYPE_SHIFT)
cdef unsigned int HANDTYPE_VALUE_TWOPAIR = ((<unsigned int>2) << HANDTYPE_SHIFT)
cdef unsigned int HANDTYPE_VALUE_PAIR = ((<unsigned int>1) << HANDTYPE_SHIFT)
cdef unsigned int HANDTYPE_VALUE_HIGHCARD = ((<unsigned int>0) << HANDTYPE_SHIFT)

#@cython.profile(True)
@cython.boundscheck(False) 
@cython.wraparound(False)
cdef unsigned int cy_evaluate(unsigned long long cards, unsigned int num_cards) nogil:
    """
    7-card evaluation function based on Keith Rule's port of PokerEval.
    Pure Python: 20000 calls in 0.176 seconds (113636 calls/sec)
    Cython: 20000 calls in 0.044 seconds (454545 calls/sec)
    """
    cdef unsigned int retval = 0, four_mask, three_mask, two_mask
    
    cdef unsigned int sc = <unsigned int>((cards >> (CLUB_OFFSET)) & 0x1fffUL)
    cdef unsigned int sd = <unsigned int>((cards >> (DIAMOND_OFFSET)) & 0x1fffUL)
    cdef unsigned int sh = <unsigned int>((cards >> (HEART_OFFSET)) & 0x1fffUL)
    cdef unsigned int ss = <unsigned int>((cards >> (SPADE_OFFSET)) & 0x1fffUL)
    
    cdef unsigned int ranks = sc | sd | sh | ss
    cdef unsigned int n_ranks = N_BITS_TABLE[ranks]
    cdef unsigned int n_dups = <unsigned int>(num_cards - n_ranks)
    
    cdef unsigned int st, t, kickers, second, tc, top
    
    if n_ranks >= 5:
        if N_BITS_TABLE[ss] >= 5:
            if STRAIGHT_TABLE[ss] != 0:
                return HANDTYPE_VALUE_STRAIGHTFLUSH + <unsigned int>(STRAIGHT_TABLE[ss] << TOP_CARD_SHIFT)
            else:
                retval = HANDTYPE_VALUE_FLUSH + TOP_FIVE_CARDS_TABLE[ss]
        elif N_BITS_TABLE[sc] >= 5:
            if STRAIGHT_TABLE[sc] != 0:
                return HANDTYPE_VALUE_STRAIGHTFLUSH + <unsigned int>(STRAIGHT_TABLE[sc] << TOP_CARD_SHIFT)
            else:
                retval = HANDTYPE_VALUE_FLUSH + TOP_FIVE_CARDS_TABLE[sc]
        elif N_BITS_TABLE[sd] >= 5:
            if STRAIGHT_TABLE[sd] != 0:
                return HANDTYPE_VALUE_STRAIGHTFLUSH + <unsigned int>(STRAIGHT_TABLE[sd] << TOP_CARD_SHIFT)
            else:
                retval = HANDTYPE_VALUE_FLUSH + TOP_FIVE_CARDS_TABLE[sd]
        elif N_BITS_TABLE[sh] >= 5:
            if STRAIGHT_TABLE[sh] != 0:
                return HANDTYPE_VALUE_STRAIGHTFLUSH + <unsigned int>(STRAIGHT_TABLE[sh] << TOP_CARD_SHIFT)
            else:
                retval = HANDTYPE_VALUE_FLUSH + TOP_FIVE_CARDS_TABLE[sh]
        else:
            st = STRAIGHT_TABLE[ranks]
            if st != 0:
                retval = HANDTYPE_VALUE_STRAIGHT + (st << TOP_CARD_SHIFT)

        if retval != 0 and n_dups < 3:
            return retval

    if n_dups == 0:
        return HANDTYPE_VALUE_HIGHCARD + TOP_FIVE_CARDS_TABLE[ranks]
    elif n_dups == 1:
        two_mask = ranks ^ (sc ^ sd ^ sh ^ ss)
        retval = <unsigned int>(HANDTYPE_VALUE_PAIR + (TOP_CARD_TABLE[two_mask] << TOP_CARD_SHIFT))
        t = ranks ^ two_mask
        kickers = (TOP_FIVE_CARDS_TABLE[t] >> CARD_WIDTH) & ~FIFTH_CARD_MASK
        retval += kickers
        return retval
    elif n_dups == 2:
        two_mask = ranks ^ (sc ^ sd ^ sh ^ ss)
        if two_mask != 0:
            t = ranks ^ two_mask
            retval = <unsigned int>(HANDTYPE_VALUE_TWOPAIR
                + (TOP_FIVE_CARDS_TABLE[two_mask]
                & (TOP_CARD_MASK | SECOND_CARD_MASK))
                + (TOP_CARD_TABLE[t] << THIRD_CARD_SHIFT))
            return retval
        else:
            three_mask = ((sc & sd) | (sh & ss)) & ((sc & sh) | (sd & ss))
            retval = <unsigned int>(HANDTYPE_VALUE_TRIPS + (TOP_CARD_TABLE[three_mask] << TOP_CARD_SHIFT))
            t = ranks ^ three_mask
            second = TOP_CARD_TABLE[t]
            retval += (second << SECOND_CARD_SHIFT)
            t ^= (1U << <int>second)
            retval += <unsigned int>(TOP_CARD_TABLE[t] << THIRD_CARD_SHIFT)
            return retval
    else:
        four_mask = sh & sd & sc & ss
        if four_mask != 0:
            tc = TOP_CARD_TABLE[four_mask]
            retval = <unsigned int>(HANDTYPE_VALUE_FOUR_OF_A_KIND
                + (tc << TOP_CARD_SHIFT)
                + ((TOP_CARD_TABLE[ranks ^ (1U << <int>tc)]) << SECOND_CARD_SHIFT))
            return retval
        two_mask = ranks ^ (sc ^ sd ^ sh ^ ss)
        if N_BITS_TABLE[two_mask] != n_dups:
            three_mask = ((sc & sd) | (sh & ss)) & ((sc & sh) | (sd & ss))
            retval = HANDTYPE_VALUE_FULLHOUSE
            tc = TOP_CARD_TABLE[three_mask]
            retval += (tc << TOP_CARD_SHIFT)
            t = (two_mask | three_mask) ^ (1U << <int>tc)
            retval += <unsigned int>(TOP_CARD_TABLE[t] << SECOND_CARD_SHIFT)
            return retval
        if retval != 0:
            return retval
        else:
            retval = HANDTYPE_VALUE_TWOPAIR
            top = TOP_CARD_TABLE[two_mask]
            retval += (top << TOP_CARD_SHIFT)
            second = TOP_CARD_TABLE[two_mask ^ (1 << <int>top)]
            retval += (second << SECOND_CARD_SHIFT)
            retval += <unsigned int>((TOP_CARD_TABLE[ranks ^ (1U << <int>top) ^ (1 << <int>second)]) << THIRD_CARD_SHIFT)
            return retval


######################
# https://github.com/obachem/kmc2/blob/master/kmc2.pyx
# 
######################
def kmc2(X, k, chain_length=200, afkmc2=True, random_state=None, weights=None):
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

    # Handle input
    X = check_array(X, accept_sparse="csr", dtype=numpy.float64, order="C")    
    sparse = not isinstance(X, numpy.ndarray)
    if weights is None:
        weights = numpy.ones(X.shape[0], dtype=numpy.float64)
    if random_state is None or isinstance(random_state, int):
        random_state = numpy.random.RandomState(random_state)
    if not isinstance(random_state, numpy.random.RandomState):
        raise ValueError("RandomState should either be a numpy.random.RandomState"
                         " instance, None or an integer to be used as seed.")
    # Initialize result
    centers = numpy.zeros((k, X.shape[1]), numpy.float64, order="C")
    # Sample first center and compute proposal
    rel_row = X[random_state.choice(X.shape[0], p=weights/weights.sum()), :]
    centers[0, :] = rel_row.todense().flatten() if sparse else rel_row
    if afkmc2:
        if(X.shape[1] == 1):
            di = numpy.reshape(euclidean_distances(X, centers[0:1, :], squared = True), newshape = (X.shape[0]))
        else:
            di = numpy.min(euclidean_distances(X, centers[0:1, :], squared=True), axis=1)*weights
        q = di/numpy.sum(di) + weights/numpy.sum(weights)  # Only the potentials
    else:
        q = numpy.copy(weights)
    # Renormalize the proposal distribution
    q = q / numpy.sum(q)

    for i in range(k-1):
        t1 = time.time()
        # Draw the candidate indices
        cand_ind = random_state.choice(X.shape[0], size=(chain_length), p=q).astype(numpy.intp)
        # Extract the proposal probabilities
        q_cand = q[cand_ind]
        # Compute pairwise distances
        dist = euclidean_distances(X[cand_ind, :], centers[0:(i+1), :], squared=True)
        # Compute potentials
        p_cand = numpy.min(dist, axis=1)*weights[cand_ind]
        # Compute acceptance probabilities
        rand_a = random_state.random_sample(size=(chain_length))
        with nogil, cython.boundscheck(False), cython.wraparound(False), cython.cdivision(True):
            # Markov chain
            for j in range(q_cand.shape[0]):
                cand_prob = p_cand[j]/q_cand[j]
                if j == 0 or curr_prob == 0.0 or cand_prob/curr_prob > rand_a[j]:
                    # Init new chain             Metropolis-Hastings step
                    curr_ind = j
                    curr_prob = cand_prob
        rel_row = X[cand_ind[curr_ind], :]
        centers[i+1, :] = rel_row.todense().flatten() if sparse else rel_row

        t2 = time.time()
        t = t2 - t1
        print('~' + str(t * (k - 1 - i)//60) + " minutes until completion. " + str(100 * (i / (k - 1)))[:4] + "% done", end = '\r')
    print()
    return centers



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
# See if a card appears twice in the hand/table combo. Can be faster.
# 

# @cython.profile(True)
@cython.boundscheck(False) 
@cython.wraparound(False)
cdef numpy.npy_bool contains_duplicates(int16[:] XX) nogil:
    cdef unsigned int count
    cdef unsigned int length
    cdef unsigned int countt = 0
    length = 7
    for count in range(length):        
        for countt in range(count+1, length):
            if(XX[count] == XX[countt]):
                return True
        
    return False


# 
# Check if the hand/board combo (XX) contains a given card (comp)
# 

# @cython.profile(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef numpy.npy_bool contains(int16[:] XX, int comp) nogil:
    cdef unsigned int count
    cdef unsigned int length = 7
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

# @cython.profile(True)
@cython.boundscheck(False) 
@cython.wraparound(False)
cdef void get_ehs_fast(int16[:] j, int16[:] twl_tiewinloss) nogil:
    cdef int16 x[45]
    cdef int16 i, k
    cdef int hero, v, c
    cdef unsigned long long mask = 0
    cdef unsigned int seven = 7
    cdef unsigned long long one = 1

    
    c = 0
    for i in range(52):
        if not contains(j, i):
            x[c] = i
            c += 1    

    mask |= one << j[2]
    mask |= one << j[3]
    mask |= one << j[4]
    mask |= one << j[5]
    mask |= one << j[6]

    hero = cy_evaluate(one << j[0] | one << j[1] | mask, seven)

    for i in range(0, 45-1):
        for k in range(i+1, 45):

            v = cy_evaluate(one << x[i] | one << x[k] | mask, seven)

            if(hero > v):
                twl_tiewinloss[1] += 1
            elif(v > hero):
                twl_tiewinloss[2] += 1
            else:
                twl_tiewinloss[0] += 1
    

# 
# All major computation is done in C. Only remaining overhead is encountered in the
# below function. For each of the (legal) C(52, 2) * C(50, 5) combinations that represent all 
# of hero's hand/table combos we make C(45, 2) comparisons with the other legal villian hands.
# The cumulative comparisons done is somewhere between (C(52, 7) * C(45, 2)) and 
# (C(52, 2) * C(52, 5) * C(45, 2)). Most of the current optimizations come in the way of
# memory management (minimizing reads/writes to existing/new locations).
# 
# Will formally calculate another time... ~ it go fast ~ but it can go a good bit faster.
# 

# @cython.profile(True)
@cython.boundscheck(False) 
@cython.wraparound(False)
def do_calc(numpy.int64_t os, int16[:, :] x, int16[:, :] y, int id):
    cdef unsigned long long int total = 0
    cdef numpy.int64_t i, j, c
    cdef double t1, t2, t
    cdef numpy.int64_t x_shape = x.shape[0]
    cdef numpy.int64_t y_shape = y.shape[0]
    
    

    z_memmap = numpy.memmap('results/river.npy', mode = 'r+', dtype = numpy.int16, shape = (x_shape * y_shape, x.shape[1] + y.shape[1] + 3), offset = os)
    cdef int16 [:, :] z_view = z_memmap[:]
    cdef numpy.ndarray[int16, ndim=1] oh = numpy.empty(7, dtype=numpy.int16)
    cdef int16 [:] oh_view = oh


    c = 0
    for i in range(x_shape):
        t1=time.time()
        for j in range(y_shape):
            oh_view[:2] = x[i]
            oh_view[2:] = y[j]
            if(not contains_duplicates(oh_view)):
                z_view[c][:7] =  oh_view
                get_ehs_fast(oh_view, z_view[c][7:])
            c += 1
        t2=time.time()
        t = t2-t1
        print('~' + str(t*(x_shape - i)//60) + ' minutes until finished. ' + str(100*(i/x_shape))[:4] + '% done     ', end = '\r')
    

# 
# 
# De Dupe Da Dataset
# 
# 

@cython.boundscheck(False) 
@cython.wraparound(False)
cpdef de_dupe_cp(int64 strt,int64 x_shape0,int64 y_shape0,int64 ndupes):
    z_mm = numpy.memmap('results/river.npy', mode = 'r+', dtype = numpy.int16, shape = (x_shape0 * y_shape0, 10))
    ND_mm = numpy.memmap('results/no_dupe_river.npy', mode = 'r+', dtype = numpy.float64, shape = (x_shape0 * (y_shape0-ndupes), 1))
    cdef int16 [:, :] z_view = z_mm[:]
    cdef numpy.float64_t [:, :] nd_view = ND_mm[:]

    cdef numpy.ndarray[int16, ndim=1] oh = numpy.empty(10, dtype=numpy.int16)
    cdef int16[:] oh_view = oh[:]

    cdef int64 c = 0
    cdef int64 cd = 0
    for i in range(x_shape0):
        for j in range(y_shape0):
            oh_view[:] = z_view[cd][:]
            if(oh_view[0] != oh_view[1]):
                nd_view[c][:] = (oh_view[8]+.5*oh_view[7]) / (oh_view[8]+oh_view[7]+oh_view[9])
                c += 1
            cd += 1
            

        
    

#
# If you want to change the metric off which you cluster, this is the function to do that in.
# Just change z to have the correct number of dimensions
#
@cython.boundscheck(False) 
@cython.wraparound(False)
def de_dupe(ndupes, x_shape0, y_shape0,new_file = False):
    if(new_file):
        z = numpy.memmap('results/no_dupe_river.npy', mode = 'w+', dtype = numpy.float64, shape = (x_shape0 * (y_shape0 - ndupes), 1))
        z.flush()

    de_dupe_cp(0, x_shape0, y_shape0, ndupes)
        
    



@cython.boundscheck(False) 
@cython.wraparound(False)
def river_ehs(n, k, threads, new_file=False):
    x = nump2(n, 2)
    y = nump2(n, k)

    if(new_file):
        z = numpy.memmap('results/river.npy', mode = 'w+', dtype = numpy.int16, shape = (y.shape[0] * x.shape[0], y.shape[1] + x.shape[1] + 3))
        z.flush()

    chunksize = len(x) // (threads-1)

    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = []
        for i in range(threads-1):
            
            strt = i * chunksize
            stp = ((i + 1) * chunksize) if i != (threads - 2) else len(x)
            
            futures.append(executor.submit(do_calc, strt * y.shape[0] * 10 * 2, x[strt:stp], y, i))
        concurrent.futures.wait(futures)

        output = [f.result() for f in futures]

    