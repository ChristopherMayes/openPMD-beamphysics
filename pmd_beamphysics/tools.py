import numpy as np

def fstr(s):
    """
    Makes a fixed string for h5 files
    """
    return np.bytes_(s)



def data_are_equal(d1, d2):
    """
    Simple utility to compare data in dicts
    
    Returns True only if all keys are the same, and all np.all data are the same
    """
    
    if set(d1) != set(d2):
        return False
    
    for k in d1:
        if not np.all(d1[k]==d2[k]):
            return False
        
    return True




#-----------------------------------------
# HDF5 utilities



def decode_attr(a):
    """
    Decodes:
        ASCII strings and arrays of them to str and arrays of str
        single-length arrays to scalar (Bmad writes this)
        
    """
    if isinstance(a, bytes):
        return a.decode('utf-8')
    
    if isinstance(a, np.ndarray):
        if a.dtype.type is np.bytes_:
            a = a.astype(str)
        if len(a) == 1:
            return a[0]
    
    return a  

def decode_attrs(attrs):
    return {k:decode_attr(v) for k,v in attrs.items()}


def encode_attr(a):
    """
    Encodes attribute
    
    See the inverse function:
        decode_attr
    
    """
    
    if isinstance(a, str):
        a = fstr(a)
    
    if isinstance(a, list) or isinstance(a, tuple):
        a = np.array(a)
    
    if isinstance(a, np.ndarray):
        if a.dtype.type is np.str_:
            a = a.astype(np.bytes_)
            
    return a

def encode_attrs(attrs):
    return {k:encode_attr(v) for k,v in attrs.items()}
