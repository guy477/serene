from .._utils._utils cimport *
from .._utils.eval.eval cimport *

cdef class HashTable:
    cdef public object table
    cdef public dict to_merge
    cdef public dict to_prune

cdef bytes abstract_key(object key)
cpdef double default_double()