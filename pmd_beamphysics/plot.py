"""


"""
from pmd_beamphysics.particles import slice_statistics
from  pmd_beamphysics.units import nice_array
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np

#import matplotlib
cmap = plt.get_cmap('viridis')
cmap.set_under('white')



def slice_plot(particle_group, stat_key='sigma_x', n_slice=40, slice_key='z'):
    """
    Complete slice plotting routine. Will plot the density of the slice key on the right axis. 
    """
    
    x_key = 'mean_'+slice_key
    x_min_key = 'min_'+slice_key
    x_max_key = 'max_'+slice_key
    y_key = stat_key
    slice_dat = slice_statistics(particle_group, n_slice=n_slice, slice_key=slice_key,
                            keys=[x_key, y_key, x_min_key, x_max_key, 'charge'])
    
    slice_dat['density'] = slice_dat['charge']/ (slice_dat[x_max_key] - slice_dat[x_min_key])
    y2_key = 'density'
    fig, ax = plt.subplots()
    
    # Get nice arrays
    x, _, prex = nice_array(slice_dat[x_key])
    y, _, prey = nice_array(slice_dat[y_key])
    y2, _, prey2 = nice_array(slice_dat[y2_key])
    
    # Add prefix to units
    x_units = prex+particle_group.units(x_key)
    y_units = prey+particle_group.units(y_key)
    
    # Convert to Amps if possible
    y2_units = f'C/{particle_group.units(x_key)}'
    if y2_units == 'C/s':
        y2_units = 'A'
    y2_units = prey2+y2_units 
    
    # Labels
    ax.set_xlabel(f'{x_key} ({x_units})' )
    ax.set_ylabel(f'{y_key} ({y_units})' )
    
    # Main plot
    ax.plot(x, y, color = 'black')
    
    #ax.set_ylim(0, 1.1*ymax )

    ax2 = ax.twinx()
    ax2.set_ylabel(f'{y2_key} ({y2_units})' )
    ax2.fill_between(x, 0, y2, color='black', alpha = 0.2)  
    
    
def marginal_plot(particle_group, key1='t', key2='p', bins=100):
    """
    Density plot and projections
    
    Example:
    
        marginal_plot(P, 't', 'energy', bins=200)   
    
    """

    # Scale to nice units and get the factor, unit prefix
    x, f1, p1 = nice_array(particle_group[key1])
    y, f2, p2 = nice_array(particle_group[key2])
    w = particle_group['weight']
    
    u1 = particle_group.units(key1)
    u2 = particle_group.units(key2)
    ux = p1+u1
    uy = p2+u2
    
    labelx = f'{key1} ({ux})'
    labely = f'{key2} ({uy})'
    
    fig = plt.figure()
    
    gs = GridSpec(4,4)
    
    ax_joint =  fig.add_subplot(gs[1:4,0:3])
    ax_marg_x = fig.add_subplot(gs[0,0:3])
    ax_marg_y = fig.add_subplot(gs[1:4,3])
    #ax_info = fig.add_subplot(gs[0, 3:4])
    #ax_info.table(cellText=['a'])
    
    # Proper weighting
    ax_joint.hexbin(x, y, C=w, reduce_C_function=np.sum, gridsize=bins, cmap=cmap, vmin=1e-15)
    
    # Manual histogramming version
    #H, xedges, yedges = np.histogram2d(x, y, weights=w, bins=bins)
    #extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    #ax_joint.imshow(H.T, cmap=cmap, vmin=1e-16, origin='lower', extent=extent, aspect='auto')
    
    dx = x.ptp()/bins
    dy = y.ptp()/bins
    ax_marg_x.hist(x, weights=w/dx/f1, bins=bins, color='gray')
    ax_marg_y.hist(y, orientation="horizontal", weights=w/dy, bins=bins, color='gray')
    
    # Turn off tick labels on marginals
    plt.setp(ax_marg_x.get_xticklabels(), visible=False)
    plt.setp(ax_marg_y.get_yticklabels(), visible=False)
    
    # Set labels on joint
    ax_joint.set_xlabel(labelx)
    ax_joint.set_ylabel(labely)
    
    # Set labels on marginals
    ax_marg_x.set_ylabel(f'C/{u1}')
    ax_marg_y.set_xlabel(f'C/{uy}')
    plt.show()