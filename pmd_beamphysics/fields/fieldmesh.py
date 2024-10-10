from pmd_beamphysics.units import dimension, dimension_name, SI_symbol, pg_units

from pmd_beamphysics.readers import component_data, expected_record_unit_dimension, field_record_components, decode_attrs, field_paths, component_from_alias, load_field_attrs, component_alias

from pmd_beamphysics.writers import write_pmd_field, pmd_field_init

from pmd_beamphysics import tools

from pmd_beamphysics.plot import plot_fieldmesh_cylindrical_2d, plot_fieldmesh_cylindrical_1d, plot_fieldmesh_rectangular_1d, plot_fieldmesh_rectangular_2d

from pmd_beamphysics.interfaces.ansys import read_ansys_ascii_3d_fields
from pmd_beamphysics.interfaces.astra import write_astra_1d_fieldmap, read_astra_3d_fieldmaps, write_astra_3d_fieldmaps, astra_1d_fieldmap_data
from pmd_beamphysics.interfaces.gpt import write_gpt_fieldmesh
from pmd_beamphysics.interfaces.impact import create_impact_solrf_fieldmap_fourier, create_impact_solrf_ele
from pmd_beamphysics.interfaces.superfish import write_superfish_t7, read_superfish_t7

from pmd_beamphysics.fields.expansion import expand_fieldmesh_from_onaxis
from pmd_beamphysics.fields.conversion import fieldmesh_rectangular_to_cylindrically_symmetric_data

import functools

from h5py import File
import numpy as np
from copy import deepcopy
import os


c_light = 299792458.


#-----------------------------------------
# Classes


axis_labels_from_geometry = {
    'rectangular': ('x', 'y', 'z'),
    'cylindrical': ('r', 'theta', 'z')
}


class FieldMesh:
    """
    Class for openPMD External Field Mesh data.
    
    Initialized on on openPMD beamphysics particle group:
    
    - **h5**: open h5 handle, or str that is a file
    - **data**: raw data
        
    The required data is stored in ._data, and consists of dicts:
    
    - `'attrs'`
    - `'components'`
    
    Component data is always 3D.
    
    Initialization from openPMD-beamphysics HDF5 file:
    
    - `FieldMesh('file.h5')`
    
    Initialization from a data dict:
    
    - `FieldMesh(data=data)`
    
    Derived properties:
                
    - `.r`, `.theta`, `.z`
    - `.Br`, `.Btheta`, `.Bz`
    - `.Er`, `.Etheta`, `.Ez`
    - `.E`, `.B`
    
    - `.phase`
    - `.scale`
    - `.factor`
    
    - `.harmonic`
    - `.frequency`
    
    - `.shape`
    - `.geometry`
    - `.mins`, `.maxs`, `.deltas`
    - `.meshgrid`
    - `.dr`, `.dtheta`, `.dz`
    
    Booleans:
    
    - `.is_pure_electric`
    - `.is_pure_magnetic`
    - `.is_static`
    
    Units and labels
    
    - `.units`
    - `.axis_labels`
    
    Plotting:
    
    - `.plot`
    - `.plot_onaxis`
    
    Writers
    
    - `.write`
    - `.write_astra_1d`
    - `.write_astra_3d`
    - `.to_cylindrical`
    - `.to_astra_1d`
    - `.to_impact_solrf`
    - `.write_gpt`
    - `.write_superfish`
        
    Constructors (class methods):
    
    - `.from_ansys_ascii_3d`
    - `.from_astra_3d`
    - `.from_superfish`
    - `.from_onaxis`
    - `.expand_onaxis`
        
    
    

    """
    def __init__(self, h5=None, data=None):
    
        if h5:
            # Allow filename
            if isinstance(h5, str):
                fname = os.path.expandvars(os.path.expanduser(h5))
                assert os.path.exists(fname), f'File does not exist: {fname}'
  
                with File(fname, 'r') as hh5:
                    fp = field_paths(hh5)
                    assert len(fp) == 1, f'Number of field paths in {h5}: {len(fp)}'
                    data = load_field_data_h5(hh5[fp[0]])

            else:
                data = load_field_data_h5(h5)
        else:
            data = load_field_data_dict(data)
            
        # Internal data
        self._data = data
            
        # Aliases (Do not set these! Set via slicing: .Bz[:] = 0
        #for k in self.components:
        #    alias = component_alias[k]
        #    self.__dict__[alias] =  self.components[k]
            
           
    # Direct access to internal data        
    @property
    def attrs(self):
        return self._data['attrs']
    
    @property
    def components(self):
        return self._data['components']  
    
    @property
    def data(self):
        return self._data
    
    
    # Conveniences 
    @property
    def shape(self):
        return tuple(self.attrs['gridSize'])
    
    @property
    def geometry(self):
        return self.attrs['gridGeometry']
    
    @property
    def scale(self):
        return self.attrs['fieldScale']    
    @scale.setter
    def scale(self, val):
        self.attrs['fieldScale']  = val
        
    @property
    def phase(self):
        """
        Returns the complex argument `phi = -2*pi*RFphase`
        to multiply the oscillating field by. 
        
        Can be set. 
        """
        return -self.attrs['RFphase']*2*np.pi
    @phase.setter
    def phase(self, val):
        """
        Complex argument in radians
        """
        self.attrs['RFphase']  = -val/(2*np.pi)    
   
    @property
    def factor(self):
        """
        factor to multiply fields by, possibly complex.
        
        `factor = scale * exp(i*phase)`
        """
        return np.real_if_close(self.scale * np.exp(1j*self.phase))           
    
    @property
    def axis_labels(self):
        """
        
        """
        return axis_labels_from_geometry[self.geometry]
    
    def axis_index(self, key):
        """
        Returns axis index for a named axis label key.
        
        Examples:
        
        - `.axis_labels == ('x', 'y', 'z')`
        - `.axis_index('z')` returns `2`
        """
        for i, name in enumerate(self.axis_labels):
            if name == key:
                return i
        raise ValueError(f'Axis not found: {key}')
    
    @property
    def coord_vecs(self):
        """
        Uses gridSpacing, gridSize, and gridOriginOffset to return coordinate vectors.
        """
        return [np.linspace(x0, x1, nx) for x0, x1, nx in zip(self.mins, self.maxs, self.shape)]
        
    def coord_vec(self, key):
        """
        Gets the coordinate vector from a named axis key. 
        """
        i = self.axis_index(key)
        return np.linspace(self.mins[i], self.maxs[i], self.shape[i])
        
    @property 
    def meshgrid(self):
        """
        Usses coordinate vectors to produce a standard numpy meshgrids. 
        """
        vecs = self.coord_vecs
        return np.meshgrid(*vecs, indexing='ij')
        
    
    
    @property 
    def mins(self):
        return np.array(self.attrs['gridOriginOffset'])
    @property
    def deltas(self):
        return np.array(self.attrs['gridSpacing'])
    @property
    def maxs(self):
        return self.deltas*(np.array(self.attrs['gridSize'])-1) + self.mins      
    
    @property
    def frequency(self):
        if self.is_static:
            return 0
        else:
            return self.attrs['harmonic']*self.attrs['fundamentalFrequency']
    
    
    # Logicals
    @property
    def is_pure_electric(self):
        """
        Returns True if there are no non-zero mageneticField components
        """
        klist = [key for key in self.components if not self.component_is_zero(key)]
        return all([key.startswith('electric') for key in klist])
    # Logicals
    @property
    def is_pure_magnetic(self):
        """
        Returns True if there are no non-zero electricField components
        """
        klist = [key for key in self.components if not self.component_is_zero(key)]
        return all([key.startswith('magnetic') for key in klist])
            

        
    @property
    def is_static(self):
        return  self.attrs['harmonic'] == 0
    

    
    def component_is_zero(self, key):
        """
        Returns True if all elements in a component are zero.
        """
        a = self[key]
        return not np.any(a)
                

    # Plotting
    # TODO: more general plotting
    def plot(self, 
             component=None, 
             time=None, 
             axes=None, 
             cmap=None, 
             return_figure=False, 
             **kwargs):
        
        if self.geometry == 'cylindrical':
            
            return plot_fieldmesh_cylindrical_2d(self,
                                                 component=component,
                                                 time=time,
                                                 axes=axes,
                                                 return_figure=return_figure,
                                                 cmap=cmap, **kwargs)
        elif self.geometry == 'rectangular':

            plot_fieldmesh_rectangular_2d(self,
                                          component=component,                                        
                                          time=time,
                                          axes=axes,
                                          return_figure=return_figure,
                                          cmap=cmap, **kwargs)
            
        else:
            raise NotImplementedError(f'Geometry {self.geometry} not implemented')
    
    #@functools.wraps(plot_fieldmesh_cylindrical_1d) 
    def plot_onaxis(self, *args, **kwargs):
        if self.geometry == 'cylindrical':
            return plot_fieldmesh_cylindrical_1d(self, *args, **kwargs)
        elif self.geometry == 'rectangular':
            return plot_fieldmesh_rectangular_1d(self, *args, **kwargs)
        else:
            raise ValueError(f'Unsupported geometry for plot_onaxis: {self.geometry}')
    

    
    
    def units(self, key):
        """Returns the units of any key"""
        
        # Strip any operators
        _, key = get_operator(key)
        
        # Fill out aliases 
        if key in component_from_alias:
            key = component_from_alias[key]         
        
        return pg_units(key)    
    
    # openPMD    
    def write(self, h5, name=None):
        """
        Writes openPMD-beamphysics format to an open h5 handle, or new file if h5 is a str.
        
        """
        if isinstance(h5, str):
            fname = os.path.expandvars(os.path.expanduser(h5))
            h5 = File(fname, 'w')
            pmd_field_init(h5, externalFieldPath='/ExternalFieldPath/%T/')
            g = h5.create_group('/ExternalFieldPath/1/')
        else:
            g = h5
    
        write_pmd_field(g, self.data, name=name)   
   
    @functools.wraps(write_astra_1d_fieldmap)
    def write_astra_1d(self, filePath):      
        return  write_astra_1d_fieldmap(self, filePath)
    
    def to_astra_1d(self):
        z, fz = astra_1d_fieldmap_data(self)   
        dat = np.array([z, fz]).T 
        return {'attrs': {'type': 'astra_1d'}, 'data': dat}

    def write_astra_3d(self, common_filePath, verbose=False):      
        return  write_astra_3d_fieldmaps(self, common_filePath)
          
        
    @functools.wraps(create_impact_solrf_ele)      
    def to_impact_solrf(self, *args, **kwargs):
        return create_impact_solrf_ele(self, *args, **kwargs)
      
    def to_cylindrical(self):
        """
        Returns a new FieldMesh in cylindrical geometry.
        
        If the current geometry is rectangular, this
        will use the y=0 slice.
        
        """
        if self.geometry == 'rectangular':
            return FieldMesh(data=fieldmesh_rectangular_to_cylindrically_symmetric_data(self))
        elif self.geometry == 'cylindrical':
            return self
        else:
            raise NotImplementedError(f"geometry not implemented: {self.geometry}")
            
    
    def write_gpt(self, filePath, asci2gdf_bin=None, verbose=True):
        """
        Writes a GPT field file. 
        """
    
        return write_gpt_fieldmesh(self, filePath, asci2gdf_bin=asci2gdf_bin, verbose=verbose)
    
    # Superfish
    @functools.wraps(write_superfish_t7)
    def write_superfish(self, filePath, verbose=False):
        """
        Write a Superfish T7 file. 
        
        For static fields, a Poisson T7 file is written.
        
        For dynamic (`harmonic /= 0`) fields, a Fish T7 file is written
        """
        return write_superfish_t7(self, filePath, verbose=verbose)
            
            
            
    @classmethod
    @functools.wraps(read_superfish_t7)
    def from_superfish(cls, filename, type=None, geometry='cylindrical'):
        """
        Class method to parse a superfish T7 style file.
        """        
        data = read_superfish_t7(filename, type=type, geometry=geometry)
        c = cls(data=data)
        return c               
        

    @classmethod
    def from_ansys_ascii_3d(cls, *, 
                   efile = None,
                   hfile = None,
                   frequency = None):
        """
        Class method to return a FieldMesh from ANSYS ASCII files.
        
        The format of each file is:
        header1 (ignored)
        header2 (ignored)
        x y z re_fx im_fx re_fy im_fy re_fz im_fz 
        ...
        in C order, with oscillations as exp(i*omega*t)
        
        Parameters
        ----------
        efile: str
            Filename with complex electric field data in V/m
        
        hfile: str
            Filename with complex magnetic H field data in A/m
        
        frequency: float
            Frequency in Hz
        
        Returns
        -------
        FieldMesh
        
        """
        
        if frequency is None:
            raise ValueError("Please provide a frequency")
        
        data = read_ansys_ascii_3d_fields(efile, hfile, frequency=frequency)
        return cls(data=data)
        
        
        
    @classmethod
    def from_astra_3d(cls, common_filename, frequency=0):
        """
        Class method to parse multiple 3D astra fieldmap files,
        based on the common filename.
        """
        
        data = read_astra_3d_fieldmaps(common_filename, frequency=frequency)
        return cls(data=data)
        
    @classmethod
    def from_onaxis(cls, *,
                    z=None,
                    Bz=None,
                    Ez=None,
                    frequency=0,
                    harmonic=None,
                    eleAnchorPt = 'beginning'
                   ):
            """
            
            
            Parameters 
            ----------
            z: array
                z-coordinates. Must be regularly spaced.       
            
            Bz: array, optional
                magnetic field at r=0 in T
                Default: None        
            
            Ez: array, optional
                Electric field at r=0 in V/m
                Default: None
                
            frequency: float, optional
                fundamental frequency in Hz.
                Default: 0
                
            harmonic: int, optional
                Harmonic of the fundamental the field actually oscillates at.
                Default: 1 if frequency !=0, otherwise 0. 
                
            eleAnchorPt: str, optional
                Element anchor point.
                Should be one of 'beginning', 'center', 'end'
                Default: 'beginning'
            
        
            Returns
            -------
            field: FieldMesh
                Instantiated fieldmesh
            
            """
    
            # Get spacing
            nz = len(z)
            dz = np.diff(z)
            if not np.allclose(dz, dz[0]):
                raise NotImplementedError("Irregular spacing not implemented")
            dz = dz[0]    
        
            components = {}
            if Ez is not None:
                Ez = np.squeeze(np.array(Ez))
                if Ez.ndim != 1:
                    raise ValueError(f'Ez ndim = {Ez.ndim} must be 1')
                components['electricField/z'] = Ez.reshape(1,1,len(Ez))
                
            if Bz is not None:
                Bz = np.squeeze(np.array(Bz))
                if Bz.ndim != 1:
                    raise ValueError(f'Bz ndim = {Bz.ndim} must be 1')
                components['magneticField/z'] = Bz.reshape(1,1,len(Bz))            
        
            if Bz is None and Ez is None:
                raise ValueError('Please enter Ez or Bz')
        
            # Handle harmonic options
            if frequency == 0:
                harmonic = 0
            elif harmonic is None:
                harmonic = 1
        
            attrs = {'eleAnchorPt': eleAnchorPt,
             'gridGeometry': 'cylindrical',
             'axisLabels': np.array(['r', 'theta', 'z'], dtype='<U5'),
             'gridLowerBound': np.array([0, 0, 0]),
             'gridOriginOffset': np.array([ 0. ,  0. , z.min()]),
             'gridSpacing': np.array([0. , 0.   , dz]),
             'gridSize': np.array([1,  1, nz]),
             'harmonic': harmonic,
             'fundamentalFrequency': frequency,
             'RFphase': 0,
             'fieldScale': 1.0} 
            
            data = dict(attrs=attrs, components=components)
            return cls(data=data)        
        
    
    
    @functools.wraps(expand_fieldmesh_from_onaxis)
    def expand_onaxis(self, *args, **kwargs):
        return expand_fieldmesh_from_onaxis(self, *args, **kwargs)
    
    
    
        
    def __eq__(self, other):
        """
        Checks that all attributes and component internal data are the same
        """
        
        if not tools.data_are_equal(self.attrs, other.attrs):
            return False
        
        return tools.data_are_equal(self.components, other.components)
  

 #  def __setattr__(self, key, value):
 #      print('a', key)
 #      if key in component_from_alias:
 #          print('here', key)
 #          comp = component_from_alias[key]
 #          if comp in self.components:
 #              self.components[comp] = value

 #  def __getattr__(self, key):
 #      print('a')
 #      if key in component_from_alias:
 #          print('here', key)
 #          comp = component_from_alias[key]
 #          if comp in self.components:
 #              return self.components[comp]
    
    
    

    
    def scaled_component(self, key):
        """
        
        Retruns a component scaled by the complex factor
            factor = scale*exp(i*phase)
            
            
        """

        if key in self.components:
            dat = self.components[key] 
        # Aliases
        elif key in component_from_alias:
            comp = component_from_alias[key]
            if comp in self.components:
                dat = self.components[comp]   
            else:
                # Component not present, make zeros
                return np.zeros(self.shape)
        else:
            raise ValueError(f'Component not available: {key}')
        
        # Multiply by scale factor
        factor = self.factor      
        
        if factor != 1:
            return factor*dat
        else:
            return dat

    # Convenient properties
    # TODO: Automate this?
    @property
    def r(self):
        return self.coord_vec('r')
    @property
    def theta(self):
        return self.coord_vec('theta')
    @property
    def z(self):
        return self.coord_vec('z')    
    
    # Deltas
    ## cartesian
    @property
    def dx(self):
        return self.deltas[self.axis_index('x')]
    @property
    def dy(self):
        return self.deltas[self.axis_index('y')]    
    ## cylindrical
    @property
    def dr(self):
        return self.deltas[self.axis_index('r')]
    @property
    def dtheta(self):
        return self.deltas[self.axis_index('theta')]
    @property
    def dz(self):
        return self.deltas[self.axis_index('z')]  

    # Scaled components
    # TODO: Check geometry
    ## cartesian
    @property
    def Bx(self):
        return self.scaled_component('Bx') 
    @property
    def By(self):
        return self.scaled_component('By')  
    @property
    def Ex(self):
        return self.scaled_component('Ex')  
    @property
    def Ey(self):
        return self.scaled_component('Ey')         
    
    ## cylindrical
    @property
    def Br(self):
        return self.scaled_component('Br')
    @property
    def Btheta(self):
        return self.scaled_component('Btheta')
    @property
    def Bz(self):
        return self.scaled_component('Bz')
    @property
    def Er(self):
        return self.scaled_component('Er')
    @property
    def Etheta(self):
        return self.scaled_component('Etheta')
    @property
    def Ez(self):
        return self.scaled_component('Ez')    
    
 
    
    
    @property
    def B(self):
        if self.geometry=='cylindrical':
            if self.is_static:
                return np.hypot(self['Br'], self['Bz'])
            else:
                return np.abs(self['Btheta'])
        else:
            raise ValueError(f'Unknown geometry: {self.geometry}')    
          
    @property
    def E(self):
        if self.geometry=='cylindrical':
            return np.hypot(np.abs(self['Er']), np.abs(self['Ez']))
        else:
            raise ValueError(f'Unknown geometry: {self.geometry}')    
        
    def __getitem__(self, key):
        """
        Returns component data from a key
        
        If the key starts with:
        
        - `re_`
        - `im_`
        - `abs_`
        
        the appropriate numpy operator is applied.
        
        
        
        """
        
        # 
        if key in ['r', 'theta', 'z']:
            return self.coord_vec(key)
            
            
        # Raw components
        if key in self.components:
            return self.components[key]
       
        # Check for operators
        operator, key = get_operator(key)
        
        # Scaled components
        if key == 'E':
            dat = self.E
        elif key == 'B':
            dat =  self.B
        else:
            dat = self.scaled_component(key)        
                   
        if operator:
            dat = operator(dat)
            
        return dat
        

    def copy(self):
        """Returns a deep copy"""
        return deepcopy(self)          
        
    def __repr__(self):
        memloc = hex(id(self))
        return f'<FieldMesh with {self.geometry} geometry and {self.shape} shape at {memloc}>'        

    

def get_operator(key):
    """
    Check if a key starts with `re_`, `im_`, `abs_`
    
    returns operator, newkey
    """
    # Check for simple operators
    if key.startswith('re_'):
        operator = np.real
        newkey = key[3:]
    elif key.startswith('im_'):
        operator = np.imag
        newkey = key[3:]
    elif key.startswith('abs_'):
        operator = np.abs
        newkey = key[4:]            
    else:
        operator = None 
        newkey = key
    
    return operator, newkey
    
            
def load_field_data_h5(h5, verbose=True):
    """
    
    
    If `attrs['dataOrder'] == 'F'`, will transpose.
    
    If `attrs['harmonic'] == 0`, components will be cast to real by `np.real`
    
    Returns:
        data dict
    """
    data = {'components':{}}

    # Load attributes
    attrs, other = load_field_attrs(h5.attrs, verbose=verbose)
    attrs.update(other)
    data['attrs'] = attrs
    
    # Loop over records and components
    for g, comps in field_record_components.items():
        if g not in h5:
            continue
        
        # Get the full openPMD unitDimension 
        required_dim = expected_record_unit_dimension[g]
                
        for comp in comps:
            if comp not in h5[g]:
                continue
            name = g+'/'+comp
            cdat = component_data(h5[name])
                
            # Check dimensions
            dim = h5[name].attrs['unitDimension']
            assert np.all(dim == required_dim), f'{name} with dimension {required_dim} expected for {name}, found: {dim}'
            
            # Check shape
            s1 = tuple(attrs['gridSize'])
            s2 = cdat.shape
            assert s1 == s2, f'Expected shape: {s1} != found shape: {s2}'
            
            # Static fields should be real
            if attrs['harmonic'] == 0:
                cdat = np.real(cdat)
            
            # Finally set
            
            data['components'][name] = cdat        
            
    
    return data

def load_field_data_dict(data_dict, verbose=True):
    """
    Similar to load_field_data_h5, but from a dict. 
    
    This cannot do unit checking. 
    """
    
    # The output dict
    data = {}
    
    # Load attributes
    attrs, other = load_field_attrs(data_dict['attrs'], verbose=verbose)
    attrs.update(other)
    data['attrs'] = attrs    
    
    # Go through components. Allow aliases
    comp = data['components'] = {}
    for k, v in data_dict['components'].items():
        if k in component_alias:
            comp[k] = v
        elif k in component_from_alias:
            k = component_from_alias[k]
            assert k not in data
            comp[k] = v
        else:
            raise ValueError(f'Unallowed component: {k}')
    
    
    return data











