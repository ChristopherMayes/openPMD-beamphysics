"""


"""

from  pmd_beamphysics.units import nice_array, nice_scale_prefix
from pmd_beamphysics.labels import mathlabel


from .statistics import slice_statistics
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.gridspec import GridSpec
import numpy as np
from copy import copy
CMAP0 = copy(plt.get_cmap('viridis'))
CMAP0.set_under('white')

CMAP1 = copy(plt.get_cmap('plasma'))

# For field legends
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plt_histogram(a, weights=None, bins=40):
    """
    This produces the same plot as plt.hist
    
    For reference only
    
    Note that bins='auto', etc cannot be used if there are weights.
    """
    cnts, bins = np.histogram(a, weights=weights, bins=bins)
    plt.bar(bins[:-1] + np.diff(bins) / 2, cnts, np.diff(bins))

    

    
    
def slice_plot(particle_group, 
               stat_key='sigma_x',
               n_slice=40,
               slice_key='z',
               tex=True,
               **kwargs):
    """
    Complete slice plotting routine. Will plot the density of the slice key on the right axis. 
    """
    
    x_key = 'mean_'+slice_key
    y_key = stat_key
    slice_dat = slice_statistics(particle_group, n_slice=n_slice, slice_key=slice_key,
                            keys=[x_key, y_key, 'ptp_'+slice_key, 'charge'])
    
    
    slice_dat['density'] = slice_dat['charge']/ slice_dat['ptp_'+slice_key]
    y2_key = 'density'
    fig, ax = plt.subplots(**kwargs)
    
    # Get nice arrays
    x, _, prex = nice_array(slice_dat[x_key])
    y, _, prey = nice_array(slice_dat[y_key])
    y2, _, prey2 = nice_array(slice_dat[y2_key])
    
    x_units = f'{prex}{particle_group.units(x_key)}'
    y_units = f'{prey}{particle_group.units(y_key)}'    
    
    # Convert to Amps if possible
    y2_units = f'C/{particle_group.units(x_key)}'
    if y2_units == 'C/s':
        y2_units = 'A'
    y2_units = prey2+y2_units 
    
    # Labels
    

    labelx = mathlabel(slice_key, units=x_units, tex=tex)
    labely = mathlabel(y_key, units=y_units, tex=tex)    
    labely2 = mathlabel(y2_key, units=y2_units, tex=tex)        
    
    ax.set_xlabel(labelx)
    ax.set_ylabel(labely)
    
    # Main plot
    ax.plot(x, y, color = 'black')
    
    #ax.set_ylim(0, 1.1*ymax )

    ax2 = ax.twinx()
    ax2.set_ylabel(labely2)
    ax2.fill_between(x, 0, y2, color='black', alpha = 0.2)  
    
    return fig

    
    
def density_plot(particle_group, key='x', bins=None, tex=True, **kwargs):
    """
    1D density plot. Also see: marginal_plot
    
    Example:
    
        density_plot(P, 'x', bins=100)   
    
    """
    
    if not bins:
        n = len(particle_group)
        bins = int(n/100)

    # Scale to nice units and get the factor, unit prefix
    x, f1, p1 = nice_array(particle_group[key])
    w = particle_group['weight']
    u1 = particle_group.units(key).unitSymbol
    ux = p1+u1
    
    # mathtext label
    labelx = mathlabel(key, units=ux, tex=tex)

    fig, ax = plt.subplots(**kwargs)
    
    hist, bin_edges = np.histogram(x, bins=bins, weights=w)
    hist_x = bin_edges[:-1] + np.diff(bin_edges) / 2
    hist_width =  np.diff(bin_edges)
    hist_y, hist_f, hist_prefix = nice_array(hist/hist_width)
    ax.bar(hist_x, hist_y, hist_width, color='grey')
    # Special label for C/s = A
    if u1 == 's':
        _, hist_prefix = nice_scale_prefix(hist_f/f1)
        ax.set_ylabel(f'{hist_prefix}A')
    else:
        ax.set_ylabel(f'{hist_prefix}C/{ux}')
    

    ax.set_xlabel(labelx)  
    
    return fig
        
def marginal_plot(particle_group, key1='t', key2='p', bins=None, tex=True, **kwargs):
    """
    Density plot and projections
    
    Example:
    
        marginal_plot(P, 't', 'energy', bins=200)   
    
    """
    
    if not bins:
        n = len(particle_group)
        bins = int(np.sqrt(n/4) )

    # Scale to nice units and get the factor, unit prefix
    x, f1, p1 = nice_array(particle_group[key1])
    y, f2, p2 = nice_array(particle_group[key2])

    w = particle_group['weight']
    
    u1 = particle_group.units(key1).unitSymbol
    u2 = particle_group.units(key2).unitSymbol
    ux = p1+u1
    uy = p2+u2
    
    # Handle labels. 
    labelx = mathlabel(key1, units=ux, tex=tex)
    labely = mathlabel(key2, units=uy, tex=tex)
        
    fig = plt.figure(**kwargs)
    
    gs = GridSpec(4,4)
    
    ax_joint =  fig.add_subplot(gs[1:4,0:3])
    ax_marg_x = fig.add_subplot(gs[0,0:3])
    ax_marg_y = fig.add_subplot(gs[1:4,3])
    #ax_info = fig.add_subplot(gs[0, 3:4])
    #ax_info.table(cellText=['a'])
    
    # Proper weighting
    ax_joint.hexbin(x, y, C=w, reduce_C_function=np.sum, gridsize=bins, cmap=CMAP0, vmin=1e-20)
    
    # Manual histogramming version
    #H, xedges, yedges = np.histogram2d(x, y, weights=w, bins=bins)
    #extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    #ax_joint.imshow(H.T, cmap=cmap, vmin=1e-16, origin='lower', extent=extent, aspect='auto')
    
    
    
    # Top histogram
    # Old method:
    #dx = x.ptp()/bins
    #ax_marg_x.hist(x, weights=w/dx/f1, bins=bins, color='gray')
    hist, bin_edges = np.histogram(x, bins=bins, weights=w)
    hist_x = bin_edges[:-1] + np.diff(bin_edges) / 2
    hist_width =  np.diff(bin_edges)
    hist_y, hist_f, hist_prefix = nice_array(hist/hist_width)
    ax_marg_x.bar(hist_x, hist_y, hist_width, color='gray')
    # Special label for C/s = A
    if u1 == 's':
        _, hist_prefix = nice_scale_prefix(hist_f/f1)
        ax_marg_x.set_ylabel(f'{hist_prefix}A')
    else:
        ax_marg_x.set_ylabel(f'{hist_prefix}C/{ux}')
    
    
    # Side histogram
    # Old method:
    #dy = y.ptp()/bins
    #ax_marg_y.hist(y, orientation="horizontal", weights=w/dy, bins=bins, color='gray')
    hist, bin_edges = np.histogram(y, bins=bins, weights=w)
    hist_x = bin_edges[:-1] + np.diff(bin_edges) / 2
    hist_width =  np.diff(bin_edges)
    hist_y, hist_f, hist_prefix = nice_array(hist/hist_width)
    ax_marg_y.barh(hist_x, hist_y, hist_width, color='gray')
    ax_marg_y.set_xlabel(f'{hist_prefix}C/{uy}')
    
    # Turn off tick labels on marginals
    plt.setp(ax_marg_x.get_xticklabels(), visible=False)
    plt.setp(ax_marg_y.get_yticklabels(), visible=False)
    
    # Set labels on joint
    ax_joint.set_xlabel(labelx)
    ax_joint.set_ylabel(labely)

    return fig    
    
    
def density_and_slice_plot(particle_group, key1='t', key2='p', stat_keys=['norm_emit_x', 'norm_emit_y'], bins=100, n_slice=30, tex=True):
    """
    Density plot and projections
    
    Example:
    
        marginal_plot(P, 't', 'energy', bins=200)   
    
    """

    # Scale to nice units and get the factor, unit prefix
    x, f1, p1 = nice_array(particle_group[key1])
    y, f2, p2 = nice_array(particle_group[key2])
    w = particle_group['weight']
    
    u1 = particle_group.units(key1).unitSymbol
    u2 = particle_group.units(key2).unitSymbol
    ux = p1+u1
    uy = p2+u2
    
    labelx = mathlabel(key1, units=ux, tex=tex)
    labely = mathlabel(key2, units=uy, tex=tex)
    
    fig, ax = plt.subplots()
    
    ax.set_xlabel(labelx)
    ax.set_ylabel(labely)
    
    # Proper weighting
    #ax_joint.hexbin(x, y, C=w, reduce_C_function=np.sum, gridsize=bins, cmap=cmap, vmin=1e-15)
    
    # Manual histogramming version
    H, xedges, yedges = np.histogram2d(x, y, weights=w, bins=bins)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    ax.imshow(H.T, cmap=CMAP0, vmin=1e-16, origin='lower', extent=extent, aspect='auto')
    
    # Slice data
    slice_dat = slice_statistics(particle_group, n_slice=n_slice, slice_key=key1,
                            keys= stat_keys + ['ptp_'+key1, 'mean_'+key1, 'charge'])
    
    slice_dat['density'] = slice_dat['charge']/ slice_dat['ptp_'+key1]
    
    # 
    ax2 = ax.twinx()
    #ax2.set_ylim(0, 1e-6)
    x2 = slice_dat['mean_'+key1] / f1
    ulist = [particle_group.units(k).unitSymbol for k in stat_keys]
    
    max2 = max([slice_dat[k].ptp() for k in stat_keys])
    
    f3, p3 = nice_scale_prefix(max2)
    

    
    u2 = ulist[0]
    assert all([u==u2 for u in ulist] )
    u2 = p3 + u2
    labely2 = mathlabel(*stat_keys, units=u2, tex=tex)    
    for k in stat_keys:
        label = mathlabel(k, units=u2, tex=tex)
        ax2.plot(x2, slice_dat[k]/f3, label=label)
    ax2.legend()     
    ax2.set_ylabel(labely2)  
    ax2.set_ylim(bottom=0)
    
    
    # Add density
    y2 = slice_dat['density']
    y2 = y2 * max2/y2.max() / f3 /2
    ax2.fill_between(x2, 0, y2, color='black', alpha = 0.1)  
    
    
    
    
    
    
#-------------------------------------
#-------------------------------------
# Fields

def plot_fieldmesh_cylindrical_2d(fm,
                                  component=None,
                                  time=None,
                                  axes=None,
                                  aspect='auto',
                                  cmap=None,
                                  return_figure=False,
                                  **kwargs):
    """
    Plots a fieldmesh component
    
    
    """
    
    assert fm.geometry == 'cylindrical'
    
    if component is None:
        if fm.is_pure_magnetic:
            component='B'
        else:
            component='E'
    
    if not axes:
        fig, ax = plt.subplots(**kwargs)
        
        
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)        
        
    if not cmap:
        cmap = CMAP1       
    
   
    unit = fm.units(component)
    
    xmin, _, zmin = fm.mins
    xmax, _, zmax = fm.maxs
    
    # plt.imshow on [r, z] will put z on the horizontal axis. 
    extent = [zmin, zmax, xmin, xmax]
    
    xlabel = 'z (m)'
    ylabel = 'r (m)'

    
    dat = fm[component][:,0,:]
    dat = np.real_if_close(dat)
    
    dmin = dat.min()
    dmax = dat.max()
    
    ax.set_aspect(aspect)
    # Need to flip for image
    ax.imshow(np.flipud(dat), extent=extent, cmap=cmap, aspect=aspect)
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    # Add legend
    llabel = f'{component} ({unit.unitSymbol})'
    
    norm = matplotlib.colors.Normalize(vmin=dmin, vmax=dmax)
    fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap),
             cax=cax, orientation='vertical', label=llabel)     
    
    
    if return_figure:
        return fig
    
    
    
    
    