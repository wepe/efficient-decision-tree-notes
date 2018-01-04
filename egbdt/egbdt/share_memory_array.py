import numpy, ctypes, multiprocessing

_ctypes_to_numpy = {
    ctypes.c_char   : numpy.dtype(numpy.uint8),
    ctypes.c_wchar  : numpy.dtype(numpy.int16),
    ctypes.c_byte   : numpy.dtype(numpy.int8),
    ctypes.c_ubyte  : numpy.dtype(numpy.uint8),
    ctypes.c_short  : numpy.dtype(numpy.int16),
    ctypes.c_ushort : numpy.dtype(numpy.uint16),
    ctypes.c_int    : numpy.dtype(numpy.int32),
    ctypes.c_uint   : numpy.dtype(numpy.uint32),
    ctypes.c_long   : numpy.dtype(numpy.int64),
    ctypes.c_ulong  : numpy.dtype(numpy.uint64),
    ctypes.c_float  : numpy.dtype(numpy.float32),
    ctypes.c_double : numpy.dtype(numpy.float64)}

_numpy_to_ctypes = dict(zip(_ctypes_to_numpy.values(),
                            _ctypes_to_numpy.keys()))


def shm_as_ndarray(mp_array, shape = None):
    '''Given a multiprocessing.Array, returns an ndarray pointing to
    the same data.'''

    # support SynchronizedArray:
    if not hasattr(mp_array, '_type_'):
        mp_array = mp_array.get_obj()

    dtype = _ctypes_to_numpy[mp_array._type_]
    result = numpy.frombuffer(mp_array, dtype)

    if shape is not None:
        result = result.reshape(shape)

    return numpy.asarray(result)


def ndarray_to_shm(array, lock = False):
    '''Generate an 1D multiprocessing.Array containing the data from
    the passed ndarray.  The data will be *copied* into shared
    memory.'''

    array1d = array.ravel(order = 'A')

    try:
        c_type = _numpy_to_ctypes[array1d.dtype]
    except KeyError:
        c_type = _numpy_to_ctypes[numpy.dtype(array1d.dtype)]

    result = multiprocessing.Array(c_type, array1d.size, lock = lock)
    shm_as_ndarray(result)[:] = array1d
    return result