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
             self.unitDimension = dimension[unitDimension]
        else:
            self.unitDimension = unitDimension

    def __str__(self):
        return self.unitSymbol
    
    def __repr__(self):
        return  f"pmd_unit('{self.unitSymbol}', {self.unitSI}, {self.unitDimension})"
        
        
dimension = {
   
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
dimension_name = {v: k for k, v in dimension.items()}

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
    'sec'        : pmd_unit('sec', 1, 'time'),
    'amp'        : pmd_unit('Amp', 1, 'current'),
    'K'          : pmd_unit('K', 1, 'temperture'),
    'mol'        : pmd_unit('mol', 1, 'mol'),
    'cd'         : pmd_unit('cd', 1, 'luminous'),
    'Coulomb'    : pmd_unit('Coulomb', 1, 'charge'),
    'charge_num' : pmd_unit('charge #', 1, 'charge'),
    'V_per_m'    : pmd_unit('V/m', 1, 'electric_field'),
    'V'          : pmd_unit('V', 1, 'electric_potential'),
    'c_light'    : pmd_unit('vel/c', c_light, 'velocity'),
    'm_per_s'    : pmd_unit('m/s', 1, 'velocity'),
    'eV'         : pmd_unit('eV', e_charge, 'energy'),
    'eV_per_c'   : pmd_unit('eV/c', e_charge/c_light, 'momentum'),
    'Tesla'      : pmd_unit('Tesla', 1, 'tesla')
    }    




