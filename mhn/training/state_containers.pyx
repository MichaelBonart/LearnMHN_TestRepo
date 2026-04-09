"""
This submodule contains functions and classes to store and convert mutation states used for training.
It also contains a function to compute an independence model that can be used as a starting point for training
a new MHN.
"""
# author(s): Stefan Vocht

from libc.stdlib cimport malloc, free, qsort
from libc.string cimport memcpy

from mhn.training.state_containers cimport State

import numpy as np
import warnings

# STATE_SIZE is defined in setup.py, 
# the maximum number n of genes the MHN can handle is 32 * STATE_SIZE


cdef int compute_max_mutation_number(int[:, :] mutation_data):
    """
    This function is used to compute the maximum number of mutations in a single sample of the data.
    """
    cdef int max_mutation_num = 0
    cdef int local_sum
    cdef int i, j
    for i in range(mutation_data.shape[0]):
        local_sum = 0
        for j in range(mutation_data.shape[1]):
            local_sum += mutation_data[i, j]

        if local_sum > max_mutation_num:
            max_mutation_num = local_sum

    return max_mutation_num


cdef void fill_states(State *states, int[:, :] mutation_data):
    """
    This function fills the given (yet empty) states with the information from mutation_data.
    """
    cdef int i, j
    cdef State *current_state
    cdef int gene_num = mutation_data.shape[1]

    for i in range(mutation_data.shape[0]):
        current_state = states + i

        for j in range(STATE_SIZE):
            current_state[0].parts[j] = 0

        for j in range(gene_num):
            if mutation_data[i, j]:
                current_state[0].parts[j >> 5] |=  1 << (j & 31)


cdef void sort_by_age(State *states, double *ages, int state_num):
    """
    Simplistic sort algorithm to sort both states and ages according to the age values.
    """
    cdef int i, j
    cdef double tmp_age
    cdef State tmp_state
    cdef bint changing = True

    for j in range(state_num):
        changing = False
        for i in range(state_num-j-1):
            if ages[i] > ages[i+1]:
                changing = True
                tmp_age = ages[i]
                tmp_state = states[i]
                ages[i] = ages[i+1]
                states[i] = states[i+1]
                ages[i+1] = tmp_age
                states[i+1] = tmp_state
        if not changing:
            break


cdef void construct_repetition_descriptor(state_container : StateContainer):
    """
    This function counts the repetitions of each state in a given StateContainer and stores this information in repetition_descriptor.
    The data array is compressed such that it no longer contains redundant rows more than once (additionally it may be reordered).
    """

    N = state_container.internal_data_size
    states = state_container.states
    repetition_descriptor = state_container.repetition_descriptor

    qsort(<void*>states, <size_t>N, sizeof(State), compare_states)

    for i in range(N):
        repetition_descriptor[i] = 1

    #count repetitions and number of unique entries in data array
    compr_data_size = 1
    for i in range(1, N):
        if compare_states(&states[i-1], &states[i]) == 0:
            repetition_descriptor[i] = 1 + repetition_descriptor[i-1] 
            repetition_descriptor[i-1] = 0
        else:
            compr_data_size += 1

    #allocate new shorter data array for unique entries only
    compr_states = <State *> malloc(compr_data_size * sizeof(State))
    
    if not compr_states:
        raise MemoryError()

    compr_repetition_descriptor = <int *> malloc(compr_data_size * sizeof(int))

    if not compr_repetition_descriptor:
        raise MemoryError()

   #exclude redundant entries in new data array
    j=0
    for i in range(N):
        if repetition_descriptor[i] != 0:
            compr_states[j] = states[i]
            compr_repetition_descriptor[j] = repetition_descriptor[i]
            j+=1

    free(state_container.states)
    free(state_container.repetition_descriptor)

    state_container.states = compr_states
    state_container.repetition_descriptor = compr_repetition_descriptor
    state_container.internal_data_size = compr_data_size


cdef int compare_states(const void* a, const void* b) nogil:
    cdef State a_v = (<State*>a)[0]
    cdef State b_v = (<State*>b)[0]

    for i in range(STATE_SIZE):
        if a_v.parts[i] == b_v.parts[i]: continue
        if a_v.parts[i] < b_v.parts[i]: return -1
        return 1
    return 0

cdef void construct_sorted_repetition_descriptor(int* repetition_descriptor, State *states, int N):
    """
    This function counts the repetitions of each row in mutation_data. 
    The repetition count of each row is stored in the returned array at the index of the row's first occurence in mutation_data.
    All other values in the returned array are zero.    
    """

    qsort(<void*>states, <size_t>N, sizeof(State), compare_states)
    for i in range(0,N):
        repetition_descriptor[i] = 1

    for i in range(N-1, 0, -1):
        if compare_states(&states[i-1], &states[i]) == 0:
            repetition_descriptor[i-1] = 1 + repetition_descriptor[i] 
            repetition_descriptor[i] = 0

cdef void fill_default_repetition_descriptor(int* repetition_descriptor, int N):
    """
    This function fills the repetition descriptor with ones.   
    """
    for i in range(0,N):
        repetition_descriptor[i] = 1



cdef class StateContainer:
    """
    This class is used as a wrapper such that the C array containing the States can be referenced in a Python script.

    It also makes sure that there aren't more than 32 mutations present in a single sample as this would break the algorithms.
    """

    def __init__(self, int[:, :] mutation_data, bint reduce_data_redundancies = True):
        """
        Args:
            mutation_data (np.ndarray): a 2D numpy array with dtype=np.int32 that contains only 0s and 1s. Rows represent samples, and columns represent events.
        """
        # the number of columns (number of genes) must not exceed 32 * STATE_SIZE
        if mutation_data.shape[1] > (32 * STATE_SIZE):
            raise ValueError(f"The number of genes present in the mutation data must not exceed {32 * STATE_SIZE}")

        self.data_size = mutation_data.shape[0]
        self.gene_num = mutation_data.shape[1]
        self.internal_data_size = self.data_size

        self.max_mutation_num = compute_max_mutation_number(mutation_data)
        if self.max_mutation_num == 0:
            warnings.warn("Your data does not contain any mutations, something went probably wrong")
        elif self.max_mutation_num > 32:
            raise ValueError("A single sample must not contain more than 32 mutations")

        self.states = <State *> malloc(self.internal_data_size * sizeof(State))

        if not self.states:
            raise MemoryError()

        fill_states(self.states, mutation_data)


        self.repetition_descriptor = <int *> malloc(self.internal_data_size * sizeof(int))

        if not self.repetition_descriptor:
            raise MemoryError()

        if reduce_data_redundancies:
            construct_repetition_descriptor(self)
        else:
            fill_default_repetition_descriptor(self.repetition_descriptor, self.internal_data_size)


    def get_data_shape(self):
        """
        Returns:
            tuple: Number of tumor samples and the number of genes stored in this object
        """
        return self.data_size, self.gene_num

    def get_max_mutation_num(self):
        """
        Returns:
            Maximum number of mutations present in a single sample.
        """
        return self.max_mutation_num

    def get_data_repetitions(self):
        """
        Returns:
            Two arrays:
                List of unique samples in this StateContainer.
                Repetition count of each given sample in dataset.
        """
        repetitions_python_array = []
        data_python_array = []
        for i in range(self.internal_data_size):
            repetitions_python_array.append( self.repetition_descriptor[i] )
            state_int=self.states[i]
            state_array = []
            for j in range(STATE_SIZE):
                state_array.extend([(state_int.parts[j]// (2**k))%2 for k in range(32)])
            data_python_array.append(state_array[:self.gene_num])

        return data_python_array, repetitions_python_array

    def compress_data(self):
        """
        This function removes the data of all samples whose assigned repetition_count is zero.
        The arrays *states and *repetition_descriptor are reallocated as (shorter) arrays.
        self.internal_data_size refers to the length of the compressed arrays, while self.data_size still refers to the original number of datapoints.
        """
        compr_data_size = 0
        for i in range(0, self.internal_data_size):
            if self.repetition_descriptor[i]!=0: compr_data_size+=1
        
        compr_states = <State *> malloc(compr_data_size * sizeof(State))
        
        if not compr_states:
            raise MemoryError()

        compr_repetition_descriptor = <int *> malloc(compr_data_size * sizeof(int))

        if not compr_repetition_descriptor:
            raise MemoryError()

        j=0
        for i in range(self.internal_data_size):
            if self.repetition_descriptor[i] != 0:
                compr_states[j] = self.states[i]
                compr_repetition_descriptor[j] = self.repetition_descriptor[i]
                j+=1

        free(self.states)
        free(self.repetition_descriptor)

        self.states = compr_states
        self.repetition_descriptor = compr_repetition_descriptor
        self.internal_data_size = compr_data_size

    def __dealloc__(self):
        free(self.states)
        free(self.repetition_descriptor)


cdef class StateAgeContainer(StateContainer):
    """
    This class is used as a wrapper like the StateContainer class, but also contains age information for each sample.
    """

    def __init__(self, int[:, :] mutation_data, double[:] ages):
        """
        Args:
            mutation_data (np.ndarray): 2D numpy array with dtype=np.int32 that contains only 0s and 1s. Rows represent samples, and columns represent events.
            ages (np.ndarray): 1D numpy array with dtype=np.double that contains the ages of the samples present in mutation_data.
        """
        super().__init__(mutation_data)
        if ages.shape[0] != self.data_size:
            raise ValueError("The number of given ages must align with the number of samples in the mutation data")
        self.state_ages = <double *> malloc(self.data_size * sizeof(double))
        if not self.state_ages:
            raise MemoryError()
        memcpy(self.state_ages, &ages[0], self.data_size * sizeof(double))
        sort_by_age(self.states, self.state_ages, ages.shape[0])

    def __dealloc__(self):
        free(self.state_ages)


def create_indep_model(StateContainer state_container):
    """
    Compute an independence model from the data stored in the StateContainer object, where the baseline hazard Theta_ii
    of each event is set to its empirical odds and the hazard ratios (off-diagonal entries) are set to exactly 1.
    The independence model is returned in logarithmic representation.

    Args:
        state_container (StateContainer): Data used to compute the independence model.

    Returns:
        np.ndarray: Independence model in logarithmic representation.
    """

    cdef int n = state_container.gene_num

    cdef int i, j
    cdef int sum_of_occurance

    theta = np.zeros((n, n))

    for i in range(n):
        sum_of_occurance = 0
        for j in range(state_container.internal_data_size):
            sum_of_occurance += ((state_container.states[j].parts[i >> 5] >> (i & 31)) & 1) * state_container.repetition_descriptor[j]

        if sum_of_occurance == 0:
            warnings.warn(f"During independence model creation: event {i} never occurs in the data, base rate will be 0")
            theta[i, i] = -1e10
        else:
            theta[i, i] = np.log(sum_of_occurance / (state_container.data_size - sum_of_occurance + 1e-10))

    return np.around(theta, decimals=2)