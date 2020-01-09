import numpy as np
cimport numpy as np
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _deformation_change_matrix(np.ndarray[np.uint8_t, ndim=2] defor_arr, np.ndarray[np.uint8_t, ndim=2] img_arr,
                     np.ndarray[np.int32_t, ndim=2] matrix_h, np.ndarray[np.int32_t, ndim=2] matrix_v):
    cdef int h, w
    cdef int new_i, new_j
    h, w = defor_arr.shape[0], defor_arr.shape[1]
    for i in range(h):
        for j in range(w):
            new_i = i + int(matrix_v[i, j])
            new_j = j + int(matrix_h[i, j])
            if (new_i >= 0 and new_j >= 0 and new_i < h and new_j < w):
                defor_arr[new_i, new_j] = img_arr[i, j]

def deformation_change_matrix(a, b, c, d):
    return _deformation_change_matrix(a, b, c, d)
