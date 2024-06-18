
"""
Simple units functionality for the openPMD beamphysics records.

For more advanced units, use a package like Pint:
    https://pint.readthedocs.io/


"""
import numpy as np
import scipy.constants

mec2 = scipy.constants.value('electron mass energy equivalent in MeV')*1e6
mpc2 = scipy.constants.value('proton mass energy equivalent in MeV')*1e6
c_light = scipy.constants.c
e_charge = scipy.constants.e
mu_0 = scipy.constants.mu_0 # Note that this is no longer 4pi*10^-7 !


class pmd_unit:
    """
    
    Params
    ------
    
    unitSymbol: Native units name
    unitSI:     Conversion factor to the the correspontign SI unit
    unitDimension: SI Base Exponents

    Base unit dimensions are defined as:    
       Base dimension  | exponents.       | SI unit
       ---------------- -----------------   -------
       length          : (1,0,0,0,0,0,0)     m
       mass            : (0,1,0,0,0,0,0)     kg
       time            : (0,0,1,0,0,0,0)     s
       current         : (0,0,0,1,0,0,0)     A
       temperture      : (0,0,0,0,1,0,0)     K
       mol             : (0,0,0,0,0,1,0)     mol
       luminous        : (0,0,0,0,0,0,1)     cd 

    Example:
        pmd_unit('eV', 1.602176634e-19, (2, 1, -2, 0, 0, 0, 0))
        defines that an eV is 1.602176634e-19 of base units m^2 kg/s^2, which is a Joule (J)
    
    If unitSI=0 (default), init with a known symbol:
        pmd_unit('T')
    returns:
        pmd_unit('T', 1, (0, 1, -2, -1, 0, 0, 0))
        
        
    Simple equalities are provided:
        u1 == u2
    Returns True if the params are all the same. 
    
    """
    def __init__(self, unitSymbol='', unitSI=0, unitDimension = (0,0,0,0,0,0,0)):
        
        # Allow to return an internally known unit
        if unitSI==0:
            if unitSymbol in known_unit:
                # Copy internals
                u =  known_unit[unitSymbol]
                unitSI= u.unitSI  
                unitDimension = u.unitDimension
            else:
                raise ValueError(f'unknown unitSymbol: {unitSymbol}')
                
        self._unitSymbol = unitSymbol       
        self._unitSI = unitSI
        if isinstance(unitDimension, str):
             self._unitDimension = DIMENSION[unitDimension]
        else:
            self._unitDimension = unitDimension
            
    @property 
    def unitSymbol(self):
        return self._unitSymbol
    @property
    def unitSI(self):
        return self._unitSI
    @property
    def unitDimension(self):
        return self._unitDimension 

    def __mul__(self, other):
        return multiply_units(self, other)
    
    def __truediv__(self, other):
        return divide_units(self, other)
    
    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)    
    
    
    def __str__(self):
        return self.unitSymbol
    
    def __repr__(self):
        return  f"pmd_unit('{self.unitSymbol}', {self.unitSI}, {self.unitDimension})"
  

def is_dimensionless(u):
    """Checks if the unit is dimensionless"""
    return u.unitDimension == (0,0,0,0,0,0,0)

def is_identity(u):
    """Checks if the unit is equivalent to 1"""
    return u.unitSI == 1 and u.unitDimension == (0,0,0,0,0,0,0)



def multiply_units(u1, u2):
    """
    Multiplies two pmd_unit symbols
    """

    if is_identity(u1):
        return u2
    if is_identity(u2):
        return u1    
    
    s1 = u1.unitSymbol
    s2 = u2.unitSymbol
    if s1==s2:
        symbol = f'({s1})^2'
    else:
        symbol = s1+'*'+s2
    d1 = u1.unitDimension
    d2 = u2.unitDimension
    dim=tuple(sum(x) for x in zip(d1, d2))
    unitSI = u1.unitSI * u2.unitSI
    
    return pmd_unit(unitSymbol=symbol, unitSI=unitSI, unitDimension=dim)     

def divide_units(u1, u2):
    """
    Divides two pmd_unit symbols : u1/u2
    """

    if is_identity(u2):
        return u1
    
    s1 = u1.unitSymbol
    s2 = u2.unitSymbol
    if s1==s2:
        symbol =  '1'
    else:
        symbol = s1+'/'+s2
    d1 = u1.unitDimension
    d2 = u2.unitDimension
    dim=tuple(a-b for a,b in zip(d1, d2))
    unitSI = u1.unitSI / u2.unitSI
    
    return pmd_unit(unitSymbol=symbol, unitSI=unitSI, unitDimension=dim)   
        
def sqrt_unit(u):
    """
    Returns the sqrt of a unit
    """
    u.unitDimension
    
    symbol = u.unitSymbol
    if symbol not in ['', '1']:
        symbol = fr'\sqrt{{ {symbol} }}'
    
    unitSI = np.sqrt(u.unitSI)
    dim = tuple( x/2 for x in u.unitDimension)
    
    return pmd_unit(unitSymbol=symbol, unitSI=unitSI, unitDimension=dim)   
        
# length mass time current temperature mol luminous
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
    'charge'             : (0,0,1,1,0,0,0),
    'electric_field'     : (1,1,-3,-1,0,0,0),
    'electric_potential' : (1,2,-3,-1,0,0,0),
    'magnetic_field'     : (0,1,-2,-1,0,0,0),    
    'velocity'           : (1,0,-1,0,0,0,0),
    'energy'             : (2,1,-2,0,0,0,0),
    'momentum'           : (1,1,-1,0,0,0,0)
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
    'magnetic_field' : 'T'
}
# Inverse
SI_name = {v: k for k, v in SI_symbol.items()}

# length mass time current temperature mol luminous
known_unit = { 
    '1'          : pmd_unit('', 1, '1'),
    'degree'     : pmd_unit('degree', np.pi/180, '1'),
    'rad'        : pmd_unit('rad', 1, '1'),
    'm'          : pmd_unit('m', 1, 'length'),
    'kg'         : pmd_unit('kg', 1, 'mass'),
    'g'          : pmd_unit('g', .001, 'mass'),
    's'          : pmd_unit('s', 1, 'time'),
    'A'          : pmd_unit('A', 1, 'current'),
    'K'          : pmd_unit('K', 1, 'temperture'),
    'mol'        : pmd_unit('mol', 1, 'mol'),
    'cd'         : pmd_unit('cd', 1, 'luminous'),
    'C'          : pmd_unit('C', 1, 'charge'),
    'charge_num' : pmd_unit('charge #', 1, 'charge'),
    'V/m'        : pmd_unit('V/m', 1, 'electric_field'),
    'V'          : pmd_unit('V', 1, 'electric_potential'),
    'c_light'    : pmd_unit('vel/c', c_light, 'velocity'),
    'm/s'        : pmd_unit('m/s', 1, 'velocity'),
    'eV'         : pmd_unit('eV', e_charge, 'energy'),
    'J'          : pmd_unit('J', 1, 'energy'),
    'eV/c'       : pmd_unit('eV/c', e_charge/c_light, 'momentum'),
    'eV/m'       : pmd_unit('eV/m', e_charge, (1, 1, -2, 0, 0, 0, 0)),
    'W'          : pmd_unit('W',       1, (2, 1, -3, 0, 0, 0, 0)),
    'W/rad^2'    : pmd_unit('W/rad^2', 1, (2, 1, -3, 0, 0, 0, 0)),       
    'W/m^2'      : pmd_unit('W/m^2',   1, (0, 1, -3, 0, 0, 0, 0)),
    'T'          : pmd_unit('T', 1, 'magnetic_field')
    } 



def unit(symbol):
    """
    Returns a pmd_unit from a known symbol.
    
    * is allowed between two known symbols: 
    """
    if symbol in known_unit:
        return known_unit[symbol]
    
    if '*' in symbol:
        subunits = [known_unit[s] for s in symbol.split('*')]
        # Require these to be in known units
        assert len(subunits) == 2, 'TODO: more complicated units'
        return multiply_units(subunits[0], subunits[1])
    
    raise ValueError(f'Unknown unit symbol: {symbol}')
        

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
    'µ'  :1e-6,
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

    if p10 < -24: # Limits of SI prefixes
        f= 1e-24
    elif p10 > 24:
        f= 1e24
    elif p10 <-1.5 or p10 > 2:
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
        x = max(np.ptp(a), abs(np.mean(a))) # Account for tiny spread
        
    fac, prefix = nice_scale_prefix( x )
    
    return a/fac, fac,  prefix


def plottable_array(x, nice=True, lim=None):
    """
    Similar to nice_array, but also considers limits for plotting
    
    Parameters
    ----------
    x: array-like
    nice: bool, default = True
        Scale array by some nice factor. 
    xlim: tuple, default = None
    
    Returns
    -------
    scaled_array: np.ndarray
    factor: float
    prefix: str
    xmin: float
    xmax : float
    
    """
    if lim is not None:
        if lim[0] is None:
            xmin = x.min()
        else:
            xmin = lim[0]
        if lim[1] is None:
            xmax = x.max()
        else:
            xmax = lim[1]            
 
    else:
        xmin = x.min()
        xmax = x.max() 
    
    if nice:
        _, factor, p1 = nice_array([xmin, xmax]) 
    else:
        factor, p1 = 1, ''
        
    return x/factor, factor, p1, xmin, xmax




# -------------------------
# Units for ParticleGroup

PARTICLEGROUP_UNITS = {}
for k in ['n_particle', 'status', 'id', 'n_alive', 'n_dead']:
    PARTICLEGROUP_UNITS[k] = unit('1')
for k in ['t']:
    PARTICLEGROUP_UNITS[k] = unit('s')
for k in ['energy', 'kinetic_energy', 'mass', 'higher_order_energy_spread', 'higher_order_energy']:
    PARTICLEGROUP_UNITS[k] = unit('eV')
for k in ['px', 'py', 'pz', 'p', 'pr', 'ptheta']:
    PARTICLEGROUP_UNITS[k] = unit('eV/c')
for k in ['x', 'y', 'z', 'r', 'Jx', 'Jy']:
    PARTICLEGROUP_UNITS[k] = unit('m')
for k in ['beta', 'beta_x', 'beta_y', 'beta_z', 'gamma', 'bunching']:    
    PARTICLEGROUP_UNITS[k] = unit('1')
for k in ['theta', 'bunching_phase']:    
    PARTICLEGROUP_UNITS[k] = unit('rad')
for k in ['charge', 'species_charge', 'weight']:
    PARTICLEGROUP_UNITS[k] = unit('C')
for k in ['average_current']:
    PARTICLEGROUP_UNITS[k] = unit('A')
for k in ['norm_emit_x', 'norm_emit_y']:
    PARTICLEGROUP_UNITS[k] = unit('m')
for k in ['norm_emit_4d']:
    PARTICLEGROUP_UNITS[k] = multiply_units(unit('m'), unit('m'))
for k in ['Lz']:
    PARTICLEGROUP_UNITS[k] = multiply_units(unit('m'), unit('eV/c'))    
for k in ['xp', 'yp']:
    PARTICLEGROUP_UNITS[k] = unit('rad')
for k in ['x_bar', 'px_bar', 'y_bar', 'py_bar']:
    PARTICLEGROUP_UNITS[k] = sqrt_unit(unit('m'))
for component in ['', 'x', 'y', 'z', 'theta', 'r']:
    PARTICLEGROUP_UNITS[f'E{component}'] = unit('V/m')
    PARTICLEGROUP_UNITS[f'B{component}'] = unit('T')    
    
# Twiss
for plane in ('x', 'y'):
    for k in ('alpha', 'etap'):
        PARTICLEGROUP_UNITS[f'twiss_{k}_{plane}'] = unit('1') 
    for k in ('beta', 'eta', 'emit', 'norm_emit'):
        PARTICLEGROUP_UNITS[f'twiss_{k}_{plane}'] = unit('m') 
    for k in ('gamma', ):
        PARTICLEGROUP_UNITS[f'twiss_{k}_{plane}'] = divide_units(unit('1'), unit('m') )
        
    

    

def pg_units(key):
    """
    Returns a str representing the units of any attribute
    """
    
    # Basic cases
    if key in PARTICLEGROUP_UNITS:
        return PARTICLEGROUP_UNITS[key]    
    
    for prefix in ['sigma_', 'mean_', 'min_', 'max_', 'ptp_', 'delta_']:
        if key.startswith(prefix):
            nkey = key[len(prefix):]
            return pg_units(nkey)
    
    if key.startswith('cov_'):
        subkeys = key.strip('cov_').split('__')
        unit0 = PARTICLEGROUP_UNITS[subkeys[0]] 

        unit1 = PARTICLEGROUP_UNITS[subkeys[1]] 
        
        return multiply_units(unit0, unit1)
   
    # Fields
    if key.startswith('electricField'):
        return unit('V/m')  
    if key.startswith('magneticField'):
        return unit('T')
    if key.startswith('bunching_phase'):
        return unit('rad')    
    if key.startswith('bunching'):
        return unit('1')
  
    
    raise ValueError(f'No known unit for: {key}')

 
# -------------------------
# Special parsers
    
def parse_bunching_str(s):
    """
    Parse a string of the on of the forms to extract the wavelength:
        'bunching_1.23e-4' 
        'bunching_1.23e-4_nm' 
        'bunching_phase_1.23e-4' 
        
    Returns
    -------
    wavelength: float
    
    """
    assert s.startswith('bunching_')
    
    # Remove bunching and phase prefixes
    s = s.replace('bunching_', '')
    s = s.replace('phase_', '')
    
    x = s.split('_')
    
    wavelength = float(x[0])
    
    if len(x) == 1:
        factor = 1
    elif len(x) == 2:
        unit = x[1]
        if unit == 'm':
            factor = 1
        elif unit == 'mm':
            factor = 1e-3
        elif unit == 'µm' or unit == 'um':
            factor = 1e-6
        elif unit == 'nm':
            factor = 1e-9
        else:
            raise ValueError(f'Unparsable unit: {unit}')
    else:
        raise ValueError(f'Cannot parse {s}')

    
    return wavelength * factor
    
    
    
# -------------------------
# h5 tools

def write_unit_h5(h5, u):
    """
    Writes an pmd_unit to an h5 handle
    """

    h5.attrs['unitSI'] = u.unitSI
    h5.attrs['unitDimension'] = u.unitDimension
    h5.attrs['unitSymbol'] = u.unitSymbol
    
def read_unit_h5(h5):
    """
    Reads unit data from an h5 handle and returns a pmd_unit object
    """
    a = h5.attrs
    
    unitSI = a['unitSI']
    unitDimension = tuple(a['unitDimension'])
    if 'unitSymbol' not in a:
        unitSymbol = 'unknown'
    else:
        unitSymbol=a['unitSymbol']
        
    return pmd_unit(unitSymbol=unitSymbol, unitSI=unitSI, unitDimension=unitDimension)


    



def read_dataset_and_unit_h5(h5, expected_unit=None, convert=True):
    """
    Reads a dataset that has openPMD unit attributes.
    
    expected_unit can be a pmd_unit object, or a known unit str. Examples: 'kg', 'J', 'eV'
    
    If expected_unit is given, will check that the units are compatible.
    
    If convert, the data will be returned with the expected_units. 
    
    
    Returns a tuple:
        np.array, pmd_unit
        
    """
    
    # Read the unit that is there. 
    u = read_unit_h5(h5)
    
    # Simple case
    if not expected_unit:
        return np.array(h5), u
    
    if isinstance(expected_unit, str):
        # Try to get unit
        expected_unit = unit(expected_unit)
        
    # Check dimensions
    du = divide_units(u, expected_unit)

    assert du.unitDimension ==  (0,0,0,0,0,0,0), 'incompatible units'
    
    if convert:
        fac = du.unitSI
        return fac*np.array(h5), expected_unit
    else:
        return np.array(h5), u

    
def write_dataset_and_unit_h5(h5, name, data, unit=None):
    """
    Writes data and pmd_unit to h5[name]
    
    See: read_dataset_and_unit_h5
    """
    h5[name] = data
    
    if unit:
        write_unit_h5(h5[name], unit)
        
        
        
    
    

    
    


    

