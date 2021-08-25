from .particles import ParticleGroup, single_particle
from .fields.fieldmesh import FieldMesh
from .readers import particle_paths
from .writers import pmd_init

from . import _version
__version__ = _version.get_versions()['version']
