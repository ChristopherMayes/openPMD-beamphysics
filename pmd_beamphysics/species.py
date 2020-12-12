#
# Simple species module. 
#
# TODO: replace with a real package
#

import scipy.constants

mec2 = scipy.constants.value('electron mass energy equivalent in MeV')*1e6
mpc2 = scipy.constants.value('proton mass energy equivalent in MeV')*1e6

e_charge = scipy.constants.e
c_light = scipy.constants.c

CHARGE_OF = {'electron': -e_charge,
            'positron': e_charge,
            'proton': e_charge}

CHARGE_STATE = {
    'electron': -1,
    'positron': 1,
    'proton': 1}


MASS_OF = {'electron': mec2,
           'positron': mec2,
           'proton': mpc2}




# Functions

def mass_of(species):
    if species in MASS_OF:
        return MASS_OF[species]
    
    raise ValueError(f'Species not available: {species}')
    
    
def charge_of(species):
    if species in CHARGE_OF:
        return CHARGE_OF[species]
    
    raise ValueError(f'Species not available: {species}')
    
def charge_state(species):
    if species in CHARGE_STATE:
        return CHARGE_STATE[species]
    
    raise ValueError(f'Species not available: {species}')    
    
    
    