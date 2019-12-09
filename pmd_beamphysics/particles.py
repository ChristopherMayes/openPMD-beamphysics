
from .units import dimension, dimension_name, SI_symbol, pg_units
from .interfaces.astra import write_astra
import numpy as np
import scipy.constants

mass_of = {'electron': 0.51099895000e6 # eV/c
              }
c_light = 299792458.
e_charge = scipy.constants.e
charge_of = {'electron': e_charge, 'positron':-e_charge}
charge_state = {'electron': -1}


#-----------------------------------------
# Classes

class ParticleGroup:
    """
    Particle Group class
    
    Initialized on on openPMD beamphysics particle group.
    
    The fundamental bunch data is stored in __dict__ with keys
        str: species
        int: n_particle
        np.array: x, px, y, py, z, pz, t, status, weight
    where:
        x, y, z are positions in units of [m]
        px, py, pz are momenta in units of [eV/c]
        t is time in [s]
        weight is the macro-charge weight in [C], used for all statistical calulations.
        
    Derived data can be computed as attributes:
        .gamma, .beta, .beta_x, .beta_y, .beta_z: relativistic factors [1].
        .energy, .kinetic_energy: energy, energy - mc2 in [eV]. 
        .p: total momentum in [eV/c]
        .mass: rest mass in [eV]
        
    Statistics of any of these are calculated with:
        .min(X)
        .max(X)
        .ptp(X)
        .avg(X)
        .std(X)
        .cov(X, Y, ...)
        with a string X as the name any of the properties above.
        
    Useful beam physics quantities are given as attributes:
        .norm_emit_x
        .norm_emit_y
        .higher_order_energy_spread
        .average_current
            
    All attributes can be accessed with brackets:
        [key]
    Additional keys are allowed for convenience:
        ['min_prop']   will return  .min('prop')
        ['max_prop']   will return  .max('prop')
        ['ptp_prop']   will return  .ptp('prop')
        ['mean_prop']  will return  .avg('prop')
        ['sigma_prop'] will return  .std('prop')
        
    Units for all attributes can be accessed by:
        .units(key)
    
    Particles are often stored at the same time (i.e. from a t-based code), 
    or with the same z position (i.e. from an s-based code.)
    Routines: 
        drift_to_z(z0)
        drift_to_t(t0)
    help to convert these. If no argument is given, particles will be drifted to the mean.
        
    
    """
    def __init__(self, h5=None, data=None):
    
        if h5:
            data = load_bunch_data(h5)
        
        self._settable_array_keys = ['x', 'px', 'y', 'py', 'z', 'pz', 't', 'status', 'weight']
        self._settable_scalar_keys = ['species', 'n_particle']
        self._settable_keys =  self._settable_array_keys + self._settable_scalar_keys                       
        for key in self._settable_keys:
            self.__dict__[key] = data[key]
        

    def units(self, key):
        return pg_units(key)
        
    @property
    def mass(self):
        """Rest mass in eV"""
        return mass_of[self.species]

    @property
    def species_charge(self):
        """Species charge in C"""
        return charge_of[self.species]
    
    @property
    def charge(self):
        return np.sum(self.weight)
    
    # Relativistic properties
    @property
    def p(self):
        """Total momemtum in eV/c"""
        return np.sqrt(self.px**2 + self.py**2 + self.pz**2) 
    @property
    def energy(self):
        """Total energy in eV"""
        return np.sqrt(self.px**2 + self.py**2 + self.pz**2 + self.mass**2) 
    @property
    def kinetic_energy(self):
        """Kinetic energy in eV"""
        return self.energy - self.mass
    
    @property
    def gamma(self):
        """Relativistic gamma"""
        return self.energy/self.mass
    @property
    def beta(self):
        """Relativistic beta"""
        return self.p/self.energy
    @property
    def beta_x(self):
        """Relativistic beta, x component"""
        return self.px/self.energy
    @property
    def beta_y(self):
        """Relativistic beta, y component"""
        return self.py/self.energy
    @property
    def beta_z(self):
        """Relativistic beta, z component"""
        return self.pz/self.energy
    
    def delta(self, key):
        """Attribute (array) relative to its mean"""
        return getattr(self, key) - self.avg(key)
    
    
    # Statistical property functions
    
    def min(self, key):
        """Minimum of any key"""
        return np.min(getattr(self, key))
    def max(self, key):
        """Maximum of any key"""
        return np.max(getattr(self, key)) 
    def ptp(self, key):
        """Peak-to-Peak = max - min of any key"""
        return np.ptp(getattr(self, key))     
        
    def avg(self, key):
        """Statistical average"""
        dat = getattr(self, key) # equivalent to self.key for accessing properties above
        return np.average(dat, weights=self.weight)
    def std(self, key):
        """Standard deviation (actually sample)"""
        dat = getattr(self, key) 
        avg_dat = self.avg(key)
        return np.sqrt(np.average( (dat - avg_dat)**2, weights=self.weight))
    def cov(self, *keys):
        """
        Covariance matrix from any properties
    
        Example: 
        P = ParticleGroup(h5)
        P.cov('x', 'px', 'y', 'py')
    
        """
        dats = np.array([ getattr(self, key) for key in keys ])
        return np.cov(dats, aweights=self.weight)
    
    # Beam statistics
    @property
    def norm_emit_x(self):
        """Normalized emittance in the x plane"""
        mat = self.cov('x', 'px')
        return  np.sqrt(mat[0,0]*mat[1,1]-mat[0,1]**2)/self.mass
    @property
    def norm_emit_y(self):       
        mat = self.cov('y', 'py')
        """Normalized emittance in the y plane"""
        return  np.sqrt(mat[0,0]*mat[1,1]-mat[0,1]**2)/self.mass
    @property
    def higher_order_energy_spread(self, order=2):
        """
        Fits a quadratic (order=2) to the Energy vs. time, subtracts it, finds the rms of the residual in eV.
        """
        best_fit_coeffs = np.polynomial.polynomial.polyfit(self.t, self.energy, order)
        best_fit = np.polynomial.polynomial.polyval(self.t, best_fit_coeffs)
        return np.std(self.energy - best_fit)        
    @property
    def average_current(self):
        """
        Simple average current in A: charge / dt, with dt =  (max_t - min_t)
        If particles are in t coordinates, will try dt = (max_z - min_z)*c_light*beta_z
        """
        dt = self.t.ptp()  # ptp 'peak to peak' is max - min
        if dt == 0:
            # must be in t coordinates. Calc with 
            dt = self.z.ptp() / (self.avg('beta_z')*c_light)
        return self.charge / dt
    
    def __getitem__(self, key):
        """
        Returns a property or statistical quantity that can be computed. 
        """
        if key.startswith('sigma_'):
            return self.std(key[6:])
        elif key.startswith('mean_'):
            return self.avg(key[5:])
        elif key.startswith('min_'):
            return self.min(key[4:])
        elif key.startswith('max_'):
            return self.max(key[4:])     
        elif key.startswith('ptp_'):
            return self.ptp(key[4:])         
        
        else:
            return getattr(self, key) 
    
    # TODO: should the user be allowed to do this?
    #def __setitem__(self, key, value):    
    #    assert key in self._settable_keyes, 'Error: you cannot set:'+str(key)
    #    
    #    if key in self._settable_array_keys:
    #        assert len(value) == self.n_particle
    #        self.__dict__[key] = value
    #    elif key == 
    #        print()
     
    # Simple 'tracking'     
    def drift(self, delta_t):
        """
        Drifts particles by time delta_t
        """
        self.x = self.x + self.beta_x * c_light * delta_t
        self.y = self.y + self.beta_y * c_light * delta_t
        self.z = self.z + self.beta_z * c_light * delta_t
        self.t = self.t + delta_t
    
    def drift_to_z(self, z=None):

        if not z:
            z = self.avg('z')
        dt = (z - self.z) / (self.beta_z * c_light)
        self.drift(dt)
        # Fix z to be exactly this value
        self.z = np.full(self.n_particle, z)
        
        
    def drift_to_t(self, t=None):
        """
        Drifts all particles to the same t
        
        If no z is given, particles will be drifted to the average t
        """
        if not t:
            t = self.avg('t')
        dt = t - self.t
        self.drift(dt)
        # Fix t to be exactly this value
        self.t = np.full(self.n_particle, t)
    
    # Writers
    def write_astra(self, filePath, verbose=False):
        write_astra(self, filePath, verbose=verbose)
        
    # New constructors
    def split(self, n_chunks = 100, key='z'):
        return split_particles(self, n_chunks=n_chunks, key=key)
    
    # Resample
    def resample(self, n):
        """
        Resamples n particles.
        """
        return resample(self, n)
    
    # Internal sorting
    def _sort(self, key):
        """Sorts internal arrays by key"""
        ixlist = np.argsort(self[key])
        for k in self._settable_array_keys:
            self.__dict__[k] = self[k][ixlist]    
        
    # Operator overloading    
    def __add__(self, other):
        """
        Overloads the + operator to join particle groups.
        Simply calls join_particle_groups
        """
        return join_particle_groups(self, other)
    
    def __str__(self):
        s = f'ParticleGroup with {self.n_particle} particles with total charge {self.charge} C'
        return s


#-----------------------------------------
# helper funcion for ParticleGroup class
    
    
def load_bunch_data(h5):
    """
    Load particles into structured numpy array.
    """
    n = len(h5['position/x'])
    
    attrs = dict(h5.attrs)
    data = {}
    data['species'] = attrs['speciesType'].decode('utf-8') # String
    data['n_particle'] = int(attrs['numParticles'])
    data['total_charge'] = attrs['totalCharge']*attrs['chargeUnitSI']
    
    for key in ['x', 'px', 'y', 'py', 'z', 'pz', 't']:
        data[key] = particle_array(h5, key)
        
    if 'particleStatus' in h5:
        data['status'] = particle_array(h5, 'particleStatus')
    else:
        data['status'] = np.full(data['n_particle'], 1)
    
    # Make sure weight is populated
    if 'weight' in h5:
        weight = particle_array(h5, 'weight')
        if len(weight) == 1:
            weight = np.full(data['n_particle'], weight[0])
    else:
        weight = np.full(data['n_particle'], data['total_charge']/data['n_particle'])
    data['weight'] = weight
        
    return data

    
    
def split_particles(particle_group, n_chunks = 100, key='z'):
    """
    Splits a particle group into even chunks. Returns a list of particle groups. 
    
    Useful for creating slice statistics. 
    
    """
    
    # Sorting
    zlist = getattr(particle_group, key) 
    iz = np.argsort(zlist)

    # Split particles into chunks
    plist = []
    for chunk in np.array_split(iz, n_chunks):
        # Prepare data
        data = {}
        keys = ['x', 'px', 'y', 'py', 'z', 'pz', 't', 'status', 'weight'] 
        for k in keys:
            data[k] = getattr(particle_group, k)[chunk]
        # These should be scalars
        data['n_particle'] = len(chunk)
        data['species'] = particle_group.species
        
        # New object
        p = ParticleGroup(data=data)
        plist.append(p)
        
    return plist    
   
def resample(particle_group, n):
    """
    Resamples a ParticleGroup randomly.
    
    Returns a new ParticleGroup instance.
    
    Note that this only works if the weights are the same
    
    """
    n_old = particle_group.n_particle
    assert n <= n_old, 'Cannot supersample'
    assert len(set(particle_group.weight)) == 1, 'non-unique weights for resampling.'
    ixlist = np.random.choice(n_old, n, replace=False)
    data = {}
    for key in particle_group._settable_array_keys:
        data[key] = particle_group[key][ixlist]
    data['species'] = particle_group['species']
    data['weight'] *= n_old/n # Need to re-weight
    data['n_particle'] = n
    
    return ParticleGroup(data=data)

    
def join_particle_groups(*particle_groups):
    """
    Join particle groups. 
    
    This simply concatenates the internal particle arrays.
    
    Species must be the same
    """
    species = [pg['species'] for pg in particle_groups]
    #return species 

    species0 = species[0]
    assert all([spe == species0 for spe in species]) , 'species must be the same to join'
    
    data = {}
    for key in particle_groups[0]._settable_array_keys:
        data[key] = np.hstack([pg[key] for pg in particle_groups ])
    
    data['species'] = species0
    data['n_particle'] = np.sum( [pg['n_particle'] for pg in particle_groups]) 
    
    return ParticleGroup(data=data)    
    
    
    
def slice_statistics(particle_group,  keys=['mean_z'], n_slice=40, slice_key='z'):
    """
    Slices a particle group into n slices and returns statistics from each sliced defined in keys. 
    
    These statistics should be scalar floats for now.
    
    Any key can be used to slice on. 
    
    """
    sdat = {}
    for k in keys:
        sdat[k] = np.empty(n_slice)
    for i, pg in enumerate(particle_group.split(n_slice, key=slice_key)):
        for k in keys:
            sdat[k][i] = pg[k]
            
    return sdat
    


        




#-----------------------------------------
# Records, components, units



particle_record_components = {
    'branchIndex':None,
    'chargeState':None,
    'electricField':['x', 'y', 'z'],
    'elementIndex':None,
    'magneticField':['x', 'y', 'z'],
    'locationInElement': None,
    'momentum':['x', 'y', 'z'],
    'momentumOffset':['x', 'y', 'z'],
    'photonPolarizationAmplitude':['x', 'y'],
    'photonPolarizationPhase':['x', 'y'],
    'sPosition':None,
    'totalMomentum':None,
    'totalMomentumOffset':None,
    #'particleCoordinatesToGlobalTransformation': ??
    'particleStatus':None,
    'pathLength':None,
    'position':['x', 'y', 'z'],
    'positionOffset':['x', 'y', 'z'],
    'spin':['x', 'y', 'z', 'theta', 'phi', 'psi'],
    'time':None,
    'timeOffset':None,
    'velocity':['x', 'y', 'z'],
    'weight':None
}
"""
Expected unit dimensions for paricle records
"""
particle_record_unit_dimension = {
    'branchIndex':dimension('1'),
    'chargeState':dimension('1'),
    'electricField':dimension('electric_field'),
    'elementIndex':dimension('1'),
    'magneticField':dimension('tesla'),
    'locationInElement': dimension('1'),
    'momentum':dimension('momentum'),
    'momentumOffset':dimension('momentum'),
    'photonPolarizationAmplitude':dimension('electric_field'),
    'photonPolarizationPhase':dimension('1'),
    'sPosition':dimension('length'),
    'totalMomentum':dimension('momentum'),
    'totalMomentumOffset':dimension('momentum'),
    #'particleCoordinatesToGlobalTransformation': ??
    'particleStatus':dimension('1'),
    'pathLength':dimension('length'),
    'position':dimension('length'),
    'positionOffset':dimension('length'),
    'spin':dimension('1'),
    'time':dimension('time'),
    'timeOffset':dimension('time'),
    'velocity':dimension('velocity'),
    'weight':dimension('1')
}







"""
Convenient aliases for components
"""
component_alias = {
    'x':'position/x',
    'y':'position/y',
    'z':'position/z',
    'px':'momentum/x',
    'py':'momentum/y',
    'pz':'momentum/z',
    't':'time',
    'weight':'weight',
    'status':'particleStatus'
}


def particle_paths(h5):
    """
    Uses the basePath and particlesPath to find where openPMD particles should be
    
    """
    basePath = h5.attrs['basePath'].decode('utf-8')
    particlesPath = h5.attrs['particlesPath'].decode('utf-8')
    path1, path2 = basePath.split('%T')
    tlist = list(h5[path1])
    paths =  [path1+t+path2+particlesPath for t in tlist]
    return paths



def is_constant_component(h5):
    """
    Constant record component should have 'value' and 'shape'
    """
    return 'value' and 'shape' in h5.attrs

def constant_component_value(h5):
    """
    Constant record component should have 'value' and 'shape'
    """
    unitSI = h5.attrs['unitSI']
    val = h5.attrs['value']
    if  unitSI == 1.0:
        return val
    else:
        return val*unitSI

def component_unit_dimension(h5):
    """
    Return the unit dimension tuple
    """
    return tuple(h5.attrs['unitDimension'])
    
def component_data(h5, slice = slice(None), unit_factor=1.0):
    """
    Returns a numpy array from an h5 component.
    
    Determines wheter a component has constant data, or array data, and returns that. 
    
    An optional slice allows parts of the array to be retrieved. 
    
    Unit factor is an addition factor to convert from SI units to output units. 
    
    """

    
    # look for unitSI factor. 
    if 'unitSI' in h5.attrs:
        factor = h5.attrs['unitSI']
    else:
        factor = 1.0
      
    # Additional conversion factor
    if unit_factor:
        factor *= unit_factor
        
    if is_constant_component(h5):
        dat = np.full(h5.attrs['shape'], h5.attrs['value'])
    else:
        # Retrieve dataset
        dat = h5[slice]

    if factor != 1.0:
        dat *= factor
        
    return dat


def offset_component_name(component_name):
    """
    Many components can also have an offset, as in:
    
        position/x
        positionOffset/c

    Return the appropriate name.
    """
    x = component_name.split('/')
    if len(x) == 1:
        return x[0]+'Offset'
    else:
        return x[0]+'Offset/'+x[1]


def particle_array(h5, component, slice=slice(None), include_offset=True):
    """
    Main routine to return particle arrays in fixed units.
    All units are SI except momentum, which will be in eV/c. 
    
    Example:
        particle_array(h5['data/00001/particles/'], 'px')
        Will return the momentum/x + momentumOffset/x in eV/c. 
        
        
    """

    if component in component_alias:
        component = component_alias[component]

    if component in ['momentum/x', 'momentum/y', 'momentum/z']:
        unit_factor = (299792458. / 1.60217662e-19  ) # convert J/(m/s) to eV/c
    else:
        unit_factor = 1.0

    # Get data
    dat = component_data(h5[component], slice = slice, unit_factor=unit_factor)
        
        
    # Look for offset component
    ocomponent = offset_component_name(component)
    if include_offset and ocomponent in h5 :
        offset =  component_data(h5[ocomponent], slice = slice, unit_factor=unit_factor)
        dat += offset
        
        
    return dat
        
    


    
    
def all_components(h5):
    """
    Look for possible components in a particle group
    """
    components = []
    for record_name in h5:
        if record_name not in particle_record_components:
            continue
    
        # Look for components
        possible_components = particle_record_components[record_name]
        
        if not possible_components:
            # Record is a component
            components.append(record_name)
        else:
            g = h5[record_name]
            for cname in possible_components:
                if cname in g:
                    components.append(record_name+'/'+cname)
    
    return components



def component_str(particle_group, name):
    """
    Informational string from a component in a particle group (h5)
    """
    
    g = particle_group[name]
    record_name = name.split('/')[0]
    expected_dimension = particle_record_unit_dimension[record_name]
    this_dimension =  component_unit_dimension(g)
    dname = dimension_name(this_dimension)
    symbol = SI_symbol[dname]
    
    s = name+' '
    
    if is_constant_component(g):
        val = constant_component_value(g)
        shape = g.attrs['shape']
        s += f'[constant {val} with shape {shape}]'
    else:
        s += '['+str(len(g))+' items]'
        
    if symbol != '1':
        s += f' is a {dname} with units: {symbol}'
        
    if expected_dimension != this_dimension:
        s +=', but expected units: '+ SI_symbol[dimension_name(this_dimension)]
    
    return s