import versioneer
from setuptools import setup, find_packages
from os import path

cur_dir = path.abspath(path.dirname(__file__))

with open(path.join(cur_dir, 'requirements.txt'), 'r') as f:
    requirements = f.read().split()



setup(
    name='openpmd-beamphysics',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(), 
    #packages = ['pmd_beamphysics'],
    packages=find_packages(exclude=['opmd_beamphysics']),  # This is the old package
    package_dir={'pmd_beamphysics':'pmd_beamphysics'},
    url='https://github.com/ChristopherMayes/openPMD-beamphysics',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    install_requires=requirements,
    include_package_data=True,
    python_requires='>=3.6'
)
