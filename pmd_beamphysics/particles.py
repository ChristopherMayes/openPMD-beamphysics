from pmd_beamphysics.units import dimension, dimension_name, SI_symbol, pg_units, c_light

from pmd_beamphysics.interfaces.astra import write_astra
from pmd_beamphysics.interfaces.bmad import write_bmad
from pmd_beamphysics.interfaces.genesis import write_genesis4_distribution, genesis2_beam_data,  write_genesis2_beam_file
from pmd_beamphysics.interfaces.gpt import write_gpt
from pmd_beamphysics.interfaces.impact import write_impact
from pmd_beamphysics.interfaces.litrack import write_litrack
from pmd_beamphysics.interfaces.lucretia import write_lucretia
from pmd_beamphysics.interfaces.opal import write_opal
from pmd_beamphysics.interfaces.elegant import write_elegant

from pmd_beamphysics.plot import density_plot, marginal_plot, slice_plot

from pmd_beamphysics.readers import particle_array, particle_paths
from pmd_beamphysics.species import charge_of, mass_of

from pmd_beamphysics.statistics import norm_emit_calc, normalized_particle_coordinate, particle_amplitude, particle_twiss_dispersion, matched_particles
from pmd_beamphysics.writers import write_pmd_bunch, pmd_init

from h5py import File
import numpy as np
from copy import deepcopy
import os


#-----------------------------------------
# Classes


class ParticleGroup:
    """
    Particle Group class
    
    Initialized on on openPMD beamphysics particle group:
        h5 = open h5 handle, or str that is a file
        data = raw data
    
    The required bunch data is stored in .data with keys
        np.array: x, px, y, py, z, pz, t, status, weight
        str: species
    where:
        x, y, z are positions in units of [m]
        px, py, pz are momenta in units of [eV/c]
        t is time in [s]
        weight is the macro-charge weight in [C], used for all statistical calulations.
        species is a proper species name: 'electron', etc. 
        
    Optional data:
        np.array: id
    where:
        id is a list of unique integers that identify the particles. 
    
        
    Derived data can be computed as attributes:
        .gamma, .beta, .beta_x, .beta_y, .beta_z: relativistic factors [1].
        .r, .theta: cylidrical coordinates [m], [1]
        .pr, .ptheta: momenta in the radial and angular coordinate directions [eV/c]
        .Lz: angular momentum about the z axis [m*eV/c]
        .energy : total energy [eV]
        .kinetic_energy: total energy - mc^2 in [eV]. 
        .higher_order_energy: total energy with quadratic fit in z or t subtracted [eV]
        .p: total momentum in [eV/c]
        .mass: rest mass in [eV]
        .xp, .yp: Slopes x' = dx/dz = dpx/dpz and y' = dy/dz = dpy/dpz [1].
        
    Normalized transvere coordinates can also be calculated as attributes:
        .x_bar, .px_bar, .y_bar, .py_bar in [sqrt(m)]
        The normalization is automatically calculated from the covariance matrix. 
        See functions in .statistics for more advanced usage.
        
        Their cooresponding amplitudes are:
        .Jx, .Jy [m]
        where Jx = (x_bar^2 + px_bar^2 )/2, 
        The momenta are normalized by the mass, so that:
            <Jx> = norm_emit_x
        and similar for y. 
        
    Statistics of any of these are calculated with:
        .min(X)
        .max(X)
        .ptp(X)
        .avg(X)
        .std(X)
        .cov(X, Y, ...)
        .histogramdd(X, Y, ..., bins=10, range=None)
        with a string X as the name any of the properties above.
        
    Useful beam physics quantities are given as attributes:
        .norm_emit_x
        .norm_emit_y
        .norm_emit_4d
        .higher_order_energy_spread
        .average_current
        
    Twiss parameters, including dispersion, for the 'x' or 'y' plane:
        .twiss(plane='x', fraction=0.95, p0C=None)
    For convenience, plane='xy' will calculate twiss for both planes.
    
    Twiss matched particles, using a simple linear transformation:
        .twiss_match(self, beta=None, alpha=None, plane='x', p0c=None, inplace=False)
    
                
    The weight is required and must sum to > 0. The sum of the weights is:
        .charge
    This can also be set:
        .charge = 1.234 # C, will rescale the .weight array
            
    All attributes can be accessed with brackets:
        [key]
    Additional keys are allowed for convenience:
        ['min_prop']   will return  .min('prop')
        ['max_prop']   will return  .max('prop')
        ['ptp_prop']   will return  .ptp('prop')
        ['mean_prop']  will return  .avg('prop')
        ['sigma_prop'] will return  .std('prop')
        ['cov_prop1__prop2'] will return .cov('prop1', 'prop2')[0,1]
        
    Units for all attributes can be accessed by:
        .units(key)
    
    Particles are often stored at the same time (i.e. from a t-based code), 
    or with the same z position (i.e. from an s-based code.)
    Routines: 
        drift_to_z(z0)
        drift_to_t(t0)
    help to convert these. If no argument is given, particles will be drifted to the mean.
    Related properties are:
        .in_t_coordinates returns True if all particles have the same t corrdinate
        .in_z_coordinates returns True if all particles have the same z corrdinate
        
        
        
    Convenient plotting is provided with: 
        .plot(...)
        .slice_plot(...)
        
        Use help(ParticleGroup.plot), etc. for usage. 
        
    
    """
    def __init__(self, h5=None, data=None):
    
    
        if h5 and data:
            # TODO:
            # Allow merging or changing some array with extra data
            raise NotImplementedError('Cannot init on both h5 and data')
    
        if h5:
            # Allow filename
            if isinstance(h5, str):
                fname = os.path.expandvars(h5)
                assert os.path.exists(fname), f'File does not exist: {fname}'
  
                with File(fname, 'r') as hh5:
                    pp = particle_paths(hh5)
                    assert len(pp) == 1, f'Number of particle paths in {h5}: {len(pp)}'
                    data = load_bunch_data(hh5[pp[0]])

            else:
                # Try dict
                data = load_bunch_data(h5)
        else:
            # Fill out data. Exclude species.
            data = full_data(data)
            species = list(set(data['species']))
            
            # Allow for empty data (len=0). Otherwise, check species.
            if len(species) >= 1:
                assert len(species) == 1, f'mixed species are not allowed: {species}'
                data['species'] = species[0]
            
        self._settable_array_keys = ['x', 'px', 'y', 'py', 'z', 'pz', 't', 'status', 'weight']
        # Optional data
        for k in ['id']:
            if k in data:
                self._settable_array_keys.append(k)  
            
        self._settable_scalar_keys = ['species']
        self._settable_keys =  self._settable_array_keys + self._settable_scalar_keys                       
        # Internal data. Only allow settable keys
        self._data = {k:data[k] for k in self._settable_keys}
        
    #-------------------------------------------------
    # Access to intrinsic coordinates
    @property
    def x(self):
        """
        x coordinate in [m]
        """
        return self._data['x']
    @x.setter
    def x(self, val):
        self._data['x'] = full_array(len(self), val)    
            
    @property
    def y(self):
        """
        y coordinate in [m]
        """
        return self._data['y']
    @y.setter
    def y(self, val):
        self._data['y'] = full_array(len(self), val)              
       
    @property
    def z(self):
        """
        z coordinate in [m]
        """
        return self._data['z']
    @z.setter
    def z(self, val):
        self._data['z'] = full_array(len(self), val)      
        
    @property
    def px(self):
        """
        px coordinate in [eV/c]
        """
        return self._data['px']
    @px.setter
    def px(self, val):
        self._data['px'] = full_array(len(self), val)      
        
    @property
    def py(self):
        """
        py coordinate in [eV/c]
        """
        return self._data['py']
    @py.setter
    def py(self, val):
        self._data['py'] = full_array(len(self), val)    
        
    @property
    def pz(self):
        """
        pz coordinate in [eV/c]
        """
        return self._data['pz']
    @pz.setter
    def pz(self, val):
        self._data['pz'] = full_array(len(self), val)    
        
    @property
    def t(self):
        """
        t coordinate in [s]
        """
        return self._data['t']
    @t.setter
    def t(self, val):
        self._data['t'] = full_array(len(self), val)       
        
    @property
    def status(self):
        """
        status coordinate in [1]
        """
        return self._data['status']
    @status.setter
    def status(self, val):
        self._data['status'] = full_array(len(self), val)  

    @property
    def weight(self):
        """
        weight coordinate in [C]
        """
        return self._data['weight']
    @weight.setter
    def weight(self, val):
        self._data['weight'] = full_array(len(self), val)            
        
    @property
    def id(self):
        """
        id integer 
        """
        if 'id' not in self._data:
            self.assign_id()      
        
        return self._data['id']
    @id.setter
    def id(self, val):
        self._data['id'] = full_array(len(self), val)            
    
    
    @property
    def species(self):
        """
        species string
        """
        return self._data['species']
    @species.setter
    def species(self, val):
        self._data['species'] = val
        
    @property
    def data(self):
        """
        Internal data dict
        """
        return self._data        
    
    #-------------------------------------------------
    # Derived data
            
    def assign_id(self):
        """
        Assigns unique ids, integers from 1 to n_particle
        
        """
        if 'id' not in self._settable_array_keys: 
            self._settable_array_keys.append('id')
        self.id = np.arange(1, self['n_particle']+1)             
    
    @property
    def n_particle(self):
        """Total number of particles. Same as len """
        return len(self)
    
    @property
    def n_alive(self):
        """Number of alive particles, defined by status == 1"""
        return len(np.where(self.status==1)[0])
    
    @property
    def n_dead(self):
        """Number of alive particles, defined by status != 1"""
        return self.n_particle - self.n_alive
    
        
    def units(self, key):
        """Returns the units of any key"""
        return pg_units(key)
        
    @property
    def mass(self):
        """Rest mass in eV"""
        return mass_of(self.species)

    @property
    def species_charge(self):
        """Species charge in C"""
        return charge_of(self.species)
    
    @property
    def charge(self):
        """Total charge in C"""
        return np.sum(self.weight)
    @charge.setter
    def charge(self, val):
        """Rescale weight array so that it sum to this value"""
        assert val >0, 'charge must be >0. This is used to weight the particles.'
        self.weight *= val/self.charge
        
    
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
    
    # Slopes. Note that these are relative to pz
    @property
    def xp(self):
        """x slope px/pz (dimensionless)"""
        return self.px/self.pz  
    @property
    def yp(self):
        """y slope py/pz (dimensionless)"""
        return self.py/self.pz    
    
    @property
    def higher_order_energy(self):
        """
        Fits a quadratic (order=2) to the Energy vs. time, and returns the energy with this subtracted. 
        """ 
        return self.higher_order_energy_calc(order=2)

    @property
    def higher_order_energy_spread(self):
        """
        Legacy syntax to compute the standard deviation of higher_order_energy.
        """
        return self.std('higher_order_energy')
     
    def higher_order_energy_calc(self, order=2):
        """
        Fits a polynmial with order `order` to the Energy vs. time, , and returns the energy with this subtracted. 
        """
        #order=2
        if self.std('z') < 1e-12:
            # must be at a screen. Use t
            t = self.t
        else:
            # All particles at the same time. Use z to calc t
            t = self.z/c_light
        energy = self.energy
        
        best_fit_coeffs = np.polynomial.polynomial.polyfit(t, energy, order)
        best_fit = np.polynomial.polynomial.polyval(t, best_fit_coeffs)
        return energy - best_fit
    
    # Polar coordinates. Note that these are centered at x=0, y=0, and not an averge center. 
    @property
    def r(self):
        """Radius in the xy plane: r = sqrt(x^2 + y^2) in m"""
        return np.hypot(self.x, self.y)
    @property    
    def theta(self):
        """Angle in xy plane: theta = arctan2(y, x) in radians"""
        return np.arctan2(self.y, self.x)
    @property
    def pr(self):
        """
        Momentum in the radial direction in eV/c
        r_hat = cos(theta) xhat + sin(theta) yhat
        pr = p dot r_hat
        """
        theta = self.theta
        return self.px * np.cos(theta)  + self.py * np.sin(theta)   
    @property    
    def ptheta(self):
        """     
        Momentum in the polar theta direction. 
        theta_hat = -sin(theta) xhat + cos(theta) yhat
        ptheta = p dot theta_hat
        Note that Lz = r*ptheta
        """
        theta = self.theta
        return -self.px * np.sin(theta)  + self.py * np.cos(theta)  
    @property    
    def Lz(self):
        """
        Angular momentum around the z axis in m*eV/c
        Lz = x * py - y * px
        """
        return self.x*self.py - self.y*self.px
    
    
    # Relativistic quantities
    @property
    def gamma(self):
        """Relativistic gamma"""
        return self.energy/self.mass
    
    @gamma.setter
    def gamma(self, val):
        beta_x = self.beta_x
        beta_y = self.beta_y
        beta_z = self.beta_z
        beta = self.beta
        gamma_new = full_array(len(self), val)
        energy_new = gamma_new * self.mass
        beta_new = np.sqrt(gamma_new**2 - 1)/gamma_new
        self._data['px'] = energy_new * beta_new * beta_x / beta
        self._data['py'] = energy_new * beta_new * beta_y / beta
        self._data['pz'] = energy_new * beta_new * beta_z / beta
        
    @property
    def beta(self):
        """Relativistic beta"""
        return self.p/self.energy
    @property
    def beta_x(self):
        """Relativistic beta, x component"""
        return self.px/self.energy
    
    @beta_x.setter
    def beta_x(self, val):
        self._data['px'] = full_array(len(self), val)*self.energy    
        
    @property
    def beta_y(self):
        """Relativistic beta, y component"""
        return self.py/self.energy
    
    @beta_y.setter
    def beta_y(self, val):
        self._data['py'] = full_array(len(self), val)*self.energy    
        
    @property
    def beta_z(self):
        """Relativistic beta, z component"""
        return self.pz/self.energy
    
    @beta_z.setter
    def beta_z(self, val):
        self._data['pz'] = full_array(len(self), val)*self.energy
    
    # Normalized coordinates for x and y
    @property 
    def x_bar(self):
        """Normalized x in units of sqrt(m)"""
        return normalized_particle_coordinate(self, 'x')
    @property     
    def px_bar(self):
        """Normalized px in units of sqrt(m)"""
        return normalized_particle_coordinate(self, 'px')    
    @property
    def Jx(self):
        """Normalized amplitude J in the x-px plane"""
        return particle_amplitude(self, 'x')
    
    @property 
    def y_bar(self):
        """Normalized y in units of sqrt(m)"""
        return normalized_particle_coordinate(self, 'y')
    @property     
    def py_bar(self):
        """Normalized py in units of sqrt(m)"""
        return normalized_particle_coordinate(self, 'py')
    @property
    def Jy(self):
        """Normalized amplitude J in the y-py plane"""
        return particle_amplitude(self, 'y')    
    
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
        if np.isscalar(dat): 
            return dat
        return np.average(dat, weights=self.weight)
    def std(self, key):
        """Standard deviation (actually sample)"""
        dat = getattr(self, key)
        if np.isscalar(dat):
            return 0
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
    
    def histogramdd(self, *keys, bins=10, range=None):
        """
        Wrapper for numpy.histogramdd, but accepts property names as keys.
        
        Computes the multidimensional histogram of keys. Internally uses weights. 
        
        Example:
            P.histogramdd('x', 'y', bins=50)
        Returns:
            np.array with shape 50x50, edge list 
        
        """
        H, edges = np.histogramdd(np.array([self[k] for k in list(keys)]).T, weights=self.weight, bins=bins, range=range)
        
        return H, edges
    
    
    
    
    # Beam statistics
    @property
    def norm_emit_x(self):
        """Normalized emittance in the x plane"""
        return norm_emit_calc(self, planes=['x'])
    @property
    def norm_emit_y(self):       
        """Normalized emittance in the x plane"""
        return norm_emit_calc(self, planes=['y'])
    @property
    def norm_emit_4d(self):       
        """Normalized emittance in the xy planes (4D)"""
        return norm_emit_calc(self, planes=['x', 'y'])    
    
    def twiss(self, plane='x', fraction=1, p0c=None):
        """
        Returns Twiss and Dispersion dict.
        
        plane can be:
            'x', 'y', 'xy'
        
        Optionally a fraction of the particles, based on amplitiude, can be specified.
        """
        d = {}
        for p in plane:
            d.update(particle_twiss_dispersion(self, plane=p, fraction=fraction, p0c=p0c))
        return d
   
    def twiss_match(self, beta=None, alpha=None, plane='x', p0c=None, inplace=False):
        """
        Returns a ParticleGroup with requested Twiss parameters.
        
        See: statistics.matched_particles
        """
        
        return matched_particles(self, beta=beta, alpha=alpha, plane=plane, inplace=inplace)
        
        
    @property
    def in_z_coordinates(self):
        """
        Returns True if all particles have the same z coordinate
        """ 
        # Check that z are all the same
        return len(np.unique(self.z)) == 1           
    
    @property
    def in_t_coordinates(self):
        """
        Returns True if all particles have the same t coordinate
        """ 
        # Check that t are all the same
        return len(np.unique(self.t)) == 1        
    
    
    
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
        Returns a property or statistical quantity that can be computed:
        P['x'] returns the x array
        P['sigmx_x'] returns the std(x) scalar
        P['norm_emit_x'] returns the norm_emit_x scalar
        
        Parts can also be given. Example: P[0:10] returns a new ParticleGroup with the first 10 elements.
        """
        
        # Allow for non-string operations: 
        if not isinstance(key, str):
            return particle_parts(self, key)
    
        if key.startswith('cov_'):
            subkeys = key[4:].split('__')
            assert len(subkeys) == 2, f'Too many properties in covariance request: {key}'
            return self.cov(*subkeys)[0,1]
        elif key.startswith('delta_'):
            return self.delta(key[6:])
        elif key.startswith('sigma_'):
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
    
    def where(self, x):
        return self[np.where(x)]
    
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
        
    def write_bmad(self, filePath, p0c=None, t_ref=0, verbose=False):
        write_bmad(self, filePath, p0c=p0c, t_ref=t_ref, verbose=verbose)        

    def write_elegant(self, filePath, verbose=False):
        write_elegant(self, filePath, verbose=verbose)            
        
    def write_genesis2_beam_file(self, filePath, n_slice=None, verbose=False):
        # Get beam columns 
        beam_columns = genesis2_beam_data(self, n_slice=n_slice)
        # Actually write the file
        write_genesis2_beam_file(filePath, beam_columns, verbose=verbose)  
        
    def write_genesis4_distribution(self, filePath, verbose=False):
        write_genesis4_distribution(self, filePath, verbose=verbose)
        
    def write_gpt(self, filePath, asci2gdf_bin=None, verbose=False):
        write_gpt(self, filePath, asci2gdf_bin=asci2gdf_bin, verbose=verbose)    
    
    def write_impact(self, filePath, cathode_kinetic_energy_ref=None, include_header=True, verbose=False):
        return write_impact(self, filePath, cathode_kinetic_energy_ref=cathode_kinetic_energy_ref,
                            include_header=include_header, verbose=verbose)          
        
    def write_litrack(self, filePath, p0c=None, verbose=False):        
        return write_litrack(self, outfile=filePath, p0c=p0c, verbose=verbose)      
        
    def write_lucretia(self, filePath, ele_name='BEGINNING', t_ref=0, stop_ix=None, verbose=False):       
        return write_lucretia(self, filePath, ele_name=ele_name, t_ref=t_ref, stop_ix=stop_ix)
        
    def write_opal(self, filePath, verbose=False, dist_type='emitted'):
        return write_opal(self, filePath, verbose=verbose, dist_type=dist_type)
    
        
    # openPMD    
    def write(self, h5, name=None):
        """
        Writes to an open h5 handle, or new file if h5 is a str.
        
        """
        if isinstance(h5, str):
            fname = os.path.expandvars(h5)
            g = File(fname, 'w')
            pmd_init(g, basePath='/', particlesPath='.' )
        else:
            g = h5
    
        write_pmd_bunch(g, self, name=name)        
        
        
    # Plotting
    # --------
    # TODO: more general plotting
    def plot(self, key1='x', key2=None, bins=None, return_figure=False, 
             tex=True, **kwargs):
        """
        1d or 2d density plot. 
        """
        
        if not key2:
            fig = density_plot(self, key=key1, bins=bins, tex=tex, **kwargs)
        else:
            fig = marginal_plot(self, key1=key1, key2=key2, bins=bins, tex=tex, **kwargs)
        
        if return_figure:
            return fig
        
    def slice_plot(self, key='sigma_x', 
                   n_slice=100,
                   slice_key=None,
                   tex=True,
                   return_figure=False,
                   **kwargs):
        """
        Slice statistics plot. 
        
        """        
                  
        if not slice_key:
            if self.in_t_coordinates:
                slice_key = 'z'
            else:
                slice_key = 't'
            
        fig = slice_plot(self, stat_key=key,
                         n_slice=n_slice,
                         slice_key=slice_key,
                         tex=tex,
                         **kwargs)
        
        if return_figure:
            return fig
        
        
    # New constructors
    def split(self, n_chunks = 100, key='z'):
        return split_particles(self, n_chunks=n_chunks, key=key)
    
    def copy(self):
        """Returns a deep copy"""
        return deepcopy(self)    
    
    
    # Operator overloading    
    
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
            self._data[k] = self[k][ixlist]    
        
    # Operator overloading    
    def __add__(self, other):
        """
        Overloads the + operator to join particle groups.
        Simply calls join_particle_groups
        """
        return join_particle_groups(self, other)
    
    # 
    def __contains__(self, item):
        """Checks internal data"""
        return True if item in self._data else False    
    
    def __len__(self):
        return len(self[self._settable_array_keys[0]])
    
    def __str__(self):
        s = f'ParticleGroup with {self.n_particle} particles with total charge {self.charge} C'
        return s

    def __repr__(self):
        memloc = hex(id(self))
        return f'<ParticleGroup with {self.n_particle} particles at {memloc}>'
                   


#-----------------------------------------
# helper functions for ParticleGroup class
    
    
def single_particle(x=0,
                   px=0,
                   y=0,
                   py=0,
                   z=0,
                   pz=0,
                   t=0,
                   weight=1,
                   status=1,
                   species='electron'):
    """
    Convenience function to make ParticleGroup with a single particle.
    
    Units:
        x, y, z: m
        px, py, pz: eV/c
        t: s
        weight: C
        status=1 => live particle
        
    """
    data = dict(x=x, px=px, y=y, py=py, z=z, pz=pz, t=t, weight=weight, status=status, species=species)
    return ParticleGroup(data=data)
    
    
def centroid(particle_group: ParticleGroup) -> ParticleGroup:
    """
    Convenience function to return a single particle representing
    the average of all coordinates. Only considers live particles.
    
    """
    good = particle_group.status == 1
    pg = particle_group[good]
    data = {key:pg.avg(key) for key in ['x', 'px', 'y', 'py', 'z', 'pz', 't']}
    data['species'] = pg.species
    data['weight'] = pg.charge
    data['status'] = 1
    return ParticleGroup(data=data)  
    
def load_bunch_data(h5):
    """
    Load particles into structured numpy array.
    """
    n = len(h5['position/x'])
    
    attrs = dict(h5.attrs)
    data = {}
    data['species'] = attrs['speciesType'].decode('utf-8') # String
    n_particle = int(attrs['numParticles'])
    data['total_charge'] = attrs['totalCharge']*attrs['chargeUnitSI']
    
    for key in ['x', 'px', 'y', 'py', 'z', 'pz', 't']:
        data[key] = particle_array(h5, key)
        
    if 'particleStatus' in h5:
        data['status'] = particle_array(h5, 'particleStatus')
    else:
        data['status'] = np.full(n_particle, 1)
    
    # Make sure weight is populated
    if 'weight' in h5:
        weight = particle_array(h5, 'weight')
        if len(weight) == 1:
            weight = np.full(n_particle, weight[0])
    else:
        weight = np.full(n_particle, data['total_charge']/n_particle)
    data['weight'] = weight
    
    # id should be a unique integer, no units
    # optional
    if 'id' in h5:
        data['id'] = h5['id'][:]
        
    return data


def full_array(n, val):
    """
    Casts a value into a full array of length n
    """
    if np.isscalar(val):
        return np.full(n, val)
    n_here = len(val)
    
    if n_here == 1:
        return np.full(n, val[0])
    elif n_here != n:
        raise ValueError(f'Length mismatch: len(val)={n_here}, but requested n={n}')
    # Cast to array
    return np.array(val)    
    
    

def full_data(data, exclude=None):
    """
    Expands keyed data into np arrays, assuring that the lengths of all items are the same. 
    
    Allows for some keys to be scalars or length 1, and fills them out with np.full.
    
    
    """
    
    full_data = {}
    scalars = {}
    for k, v in data.items():
        if np.isscalar(v):
            scalars[k] = v
        elif len(v) == 1:
            scalars[k] = v[0]
        else:
            # must be array
            full_data[k] = np.array(v)
    
    # Check for single particle
    if len(full_data) == 0:
        return {k:np.array([v]) for k, v in scalars.items()}
            
    # Array data should all have the same length
    nlist = [len(v) for _, v in full_data.items()]
    assert len(set(nlist)) == 1, f'arrays must have the same length. Found len: { {k:len(v) for k, v in full_data.items()} }'
    
    for k, v in scalars.items():
        full_data[k] = np.full(nlist[0], v)
    
    return full_data


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
        #keys = ['x', 'px', 'y', 'py', 'z', 'pz', 't', 'status', 'weight'] 
        for k in particle_group._settable_array_keys:
            data[k] = getattr(particle_group, k)[chunk]
        # These should be scalars
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
    
    return ParticleGroup(data=data)



def particle_parts(particle_group, x):
    """
    Gets parts of a ParticleGroup object. Returns a new ParticleGroup
    """
    data = {}
    for k in particle_group._settable_array_keys:
        data[k] = particle_group[k][x]

    for k in particle_group._settable_scalar_keys:
        data[k] = particle_group[k]

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
    
    

    


        




