from .hash_table cimport HashTable
from .._utils._utils cimport *

cdef class LocalManager:
    cdef public str base_path
    cdef public HashTable regret_sum
    cdef public HashTable strategy_sum
