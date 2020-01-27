
"""
Simple units functionality for the openPMD beamphysics records.

For more advanced units, use a package like Pint:
    https://pint.readthedocs.io/


"""
import scipy.constants

m_e = scipy.constants.value('electron mass energy equivalent in MeV')*1e6
m_p = scipy.constants.value('proton mass energy equivalent in MeV')*1e6
c_light = 299792458
e_charge = scipy.constants.e

import numpy as np


class pmd_unit:
    """
    
    Params
    ------
    
    unitSymbol: Native units name. EG 'eV'
    unitSI:     Conversion to SI
    unitDimension: SI Base Exponents

    
    """
    def __init__(self, unitSymbol='', unitSI=0, unitDimension = (0,0,0,0,0,0,0)):
        self.unitSymbol = unitSymbol       
        self.unitSI = unitSI
        if isinstance(unitDimension, str):
             self.unitDimension = DIMENSION[unitDimension]
        else:
            self.unitDimension = unitDimension

    def __str__(self):
        return self.unitSymbol
    
    def __repr__(self):
        return  f"pmd_unit('{self.unitSymbol}', {self.unitSI}, {self.unitDimension})"
        
        
DIMENSION = {
   
    '1'              : (0,0,0,0,0,0,0),
     # Base units
    'length'         : (1,0,0,0,0,0,0),
    'mass'           : (0,1,0,0,0,0,0),
    'time'           : (0,0,1,0,0,0,0),
    'current'        : (0,0,0,1,0,0,0),
    'temperture'     : (0,0,0,0,1,0,0),
    'mol'            : (0,0,0,0,0,1,0),
    'luminous'       : (0,0,0,0,0,0,1),
    #
    'charge'         : (0,0,1,1,0,0,0),
    'electric_field'  : (1,1,-3,-1,0,0,0),
    'electric_potential' : (1,2,-3,-1,0,0,0),
    'velocity'       : (1,0,-1,0,0,0,0),
    'energy'         : (2,1,-2,0,0,0,0),
    'momentum'       : (1,1,-1,0,0,0,0),
    'tesla'          : (0,1,-2,-1,0,0,0)
}
# Inverse
DIMENSION_NAME = {v: k for k, v in DIMENSION.items()}

def dimension(name):
    if name in DIMENSION:
        return DIMENSION[name]
    else:
        return None

def dimension_name(dim_array):
    return DIMENSION_NAME[tuple(dim_array)]

SI_symbol = {
    '1'              : '1',
    'length'         : 'm',
    'mass'           : 'kg',
    'time'           : 's',
    'current'        : 'A',
    'temperture'     : 'K',
    'mol'            : 'mol',
    'luminous'       : 'cd',
    'charge'         : 'C',
    'electric_field' : 'V/m',
    'electric_potential' : 'V',
    'velocity'       : 'm/s',
    'energy'         : 'J',
    'momentum'       : 'kg*m/s',
    'tesla'          : 'T'
}
# Inverse
SI_name = {v: k for k, v in SI_symbol.items()}



unit = { 
    '1'          : pmd_unit('', 1, '1'),
    'm'          : pmd_unit('m', 1, 'length'),
    'kg'         : pmd_unit('kg', 1, 'mass'),
    's'          : pmd_unit('s', 1, 'time'),
    'amp'        : pmd_unit('Amp', 1, 'current'),
    'K'          : pmd_unit('K', 1, 'temperture'),
    'mol'        : pmd_unit('mol', 1, 'mol'),
    'cd'         : pmd_unit('cd', 1, 'luminous'),
    'C'          : pmd_unit('C', 1, 'charge'),
    'charge_num' : pmd_unit('charge #', 1, 'charge'),
    'V/m'    : pmd_unit('V/m', 1, 'electric_field'),
    'V'          : pmd_unit('V', 1, 'electric_potential'),
    'c_light'    : pmd_unit('vel/c', c_light, 'velocity'),
    'm/s'    : pmd_unit('m/s', 1, 'velocity'),
    'eV'         : pmd_unit('eV', e_charge, 'energy'),
    'eV/c'   : pmd_unit('eV/c', e_charge/c_light, 'momentum'),
    'Tesla'      : pmd_unit('Tesla', 1, 'tesla')
    }    



# Dicts for prefixes
PREFIX_FACTOR = {
    'yocto-' :1e-24,
    'zepto-' :1e-21,
    'atto-'  :1e-18,
    'femto-' :1e-15,
    'pico-'  :1e-12,
    'nano-'  :1e-9 ,
    'micro-' :1e-6,
    'milli-' :1e-3 ,
    'centi-' :1e-2 ,
    'deci-'  :1e-1,
    'deca-'  :1e+1,
    'hecto-' :1e2  ,
    'kilo-'  :1e3  ,
    'mega-'  :1e6  ,
    'giga-'  :1e9  ,
    'tera-'  :1e12 ,
    'peta-'  :1e15 ,
    'exa-'   :1e18 ,
    'zetta-' :1e21 ,
    'yotta-' :1e24
}
# Inverse
PREFIX = dict( (v,k) for k,v in PREFIX_FACTOR.items())

SHORT_PREFIX_FACTOR = {
    'y'  :1e-24,
    'z'  :1e-21,
    'a'  :1e-18,
    'f'  :1e-15,
    'p'  :1e-12,
    'n'  :1e-9 ,
    'u'  :1e-6,
    'm'  :1e-3 ,
    'c'  :1e-2 ,
    'd'  :1e-1,
    ''   : 1,
    'da' :1e+1,
    'h'  :1e2  ,
    'k'  :1e3  ,
    'M'  :1e6  ,
    'G'  :1e9  ,
    'T'  :1e12 ,
    'P'  :1e15 ,
    'E'  :1e18 ,
    'Z'  :1e21 ,
    'Y'  :1e24
}
# Inverse
SHORT_PREFIX = dict( (v,k) for k,v in SHORT_PREFIX_FACTOR.items())




# Nice scaling

def nice_scale_prefix(scale):
    """
    Returns a nice factor and a SI prefix string 
    
    Example:
        scale = 2e-10
        
        f, u = nice_scale_prefix(scale)
        
        
    """
    
    if scale == 0:
        return 1, ''
    
    p10 = np.log10(abs(scale))

    if p10 <-2 or p10 > 2:
        f = 10**(p10 //3 *3)
    else:
        f = 1
    
    return f, SHORT_PREFIX[f]

def nice_array(a):
    """
    Returns a scaled array, the scaling, and a unit prefix
    
    Example: 
        nice_array( np.array([2e-10, 3e-10]) )
    Returns:
        (array([200., 300.]), 1e-12, 'p')
    
    """
    
    if np.isscalar(a):
        x = a
    elif len(a) == 1:
        x = a[0]
    else:
        a = np.array(a)
        x = a.ptp()
        
    fac, prefix = nice_scale_prefix( x )
    
    return a/fac, fac,  prefix




# -------------------------
# Units for ParitcleGroup

PARTICLEGROUP_UNITS = {}
for k in ['status']:
    PARTICLEGROUP_UNITS[k] = '1'
for k in ['t']:
    PARTICLEGROUP_UNITS[k] = 's'
for k in ['energy', 'kinetic_energy', 'mass', 'higher_order_energy_spread']:
    PARTICLEGROUP_UNITS[k] = 'eV'
for k in ['px', 'py', 'pz', 'p']:
    PARTICLEGROUP_UNITS[k] = 'eV/c'
for k in ['x', 'y', 'z']:
    PARTICLEGROUP_UNITS[k] = 'm' 
for k in ['beta', 'beta_x', 'beta_y', 'beta_z', 'gamma']:    
    PARTICLEGROUP_UNITS[k] = '1'
for k in ['charge', 'species_charge', 'weight']:
    PARTICLEGROUP_UNITS[k] = 'C'
for k in ['average_current']:
    PARTICLEGROUP_UNITS[k] = 'A'
for k in ['norm_emit_x', 'norm_emit_y']:
    PARTICLEGROUP_UNITS[k] = 'm*rad'

def pg_units(key):
    """
    Returns a str representing the units of any attribute
    """
    if key.startswith('sigma_'):
        return PARTICLEGROUP_UNITS[key[6:]]     
    elif key.startswith('mean_'):
        return PARTICLEGROUP_UNITS[key[5:]]
    elif key.startswith('min_'):
        return PARTICLEGROUP_UNITS[key[4:]]
    elif key.startswith('max_'):
        return PARTICLEGROUP_UNITS[key[4:]]
    else:
        return PARTICLEGROUP_UNITS[key]    
    