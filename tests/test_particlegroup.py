from pmd_beamphysics import ParticleGroup
import pytest
import numpy as np
import os

P = ParticleGroup('docs/examples/data/bmad_particles.h5')
    

ARRAY_KEYS = """
x y z px py pz t status weight id
p energy kinetic_energy xp yp higher_order_energy
r theta pr ptheta
Lz
gamma beta beta_x beta_y beta_z
x_bar px_bar Jx Jy
charge

""".split()    
    
    
SPECIAL_STATS = """
norm_emit_x norm_emit_y norm_emit_4d higher_order_energy_spread
average_current
n_alive
n_dead
""".split()
    
    
OPERATORS = ('min_', 'max_', 'sigma_', 'delta_', 'ptp_', 'mean_')    
    
@pytest.fixture(params=ARRAY_KEYS)
def array_key(request):
    return request.param

@pytest.fixture(params=OPERATORS)
def operator(request):
    return request.param

def test_operator(operator, array_key):
    key = f'{operator}{array_key}'
    P[key]
    
# This is probably uneccessary:    
# @pytest.fixture(params=array_keys)
# def array_key2(request):
#     return request.param    
#     
# def test_cov(array_key, array_key2):
#     P[f'cov_{array_key}__{array_key}']
                
   
@pytest.fixture(params=SPECIAL_STATS)
def special_stat(request):
    return request.param
        
def test_special_stat(special_stat):
    x = P[special_stat]
    assert np.isscalar(x)
        
        
def test_array_units_exist(array_key):
    P.units(array_key)

def test_special_units_exist(special_stat):
    P.units(special_stat)
    
    
def test_twiss():
    P.twiss('xy', fraction=0.95)
    
    
def test_write_reload(tmp_path):
    h5file = os.path.join(tmp_path, 'test.h5')
    P.write(h5file)
    
    # Equality and inequality
    P2 = ParticleGroup(h5file)
    assert P == P2
    
    P2.x +=1 
    assert P != P2