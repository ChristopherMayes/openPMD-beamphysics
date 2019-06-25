from .bunch_tools import bin_particles2d_h5, nice_phase_space_factor, nice_phase_space_label, particle_array, nice_phase_space_unit


from bokeh.plotting import figure#, show, output_notebook
from bokeh import palettes, colors
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.layouts import  gridplot
#output_notebook(verbose=False, hide_banner=True)
import itertools

import numpy as np

pal = palettes.Viridis[256]
white=colors.named.white
pal[0] = white # replace 0 with white




def plot_bunch_h5(h5_bunch, component1, component2, bins=20, nice=True, liveOnly=True):
    H, xedges, yedges = bin_particles2d_h5(h5_bunch, component1, component2, bins, liveOnly=liveOnly)
    xmin = min(xedges)
    xmax = max(xedges)
    ymin = min(yedges)
    ymax = max(yedges)
    
    
    if nice:
        f1 = nice_phase_space_factor[component1]
        f2 = nice_phase_space_factor[component2]
        xlabel =  nice_phase_space_label[component1]
        ylabel =  nice_phase_space_label[component2]
        xmin *= f1
        xmax *= f1
        ymin *= f2
        ymax *= f2
    else:
        xlabel = component1
        ylabel = component2
    
    # Form datasource
    dat = {'image':[H.transpose()], 'xmin':[xmin], 'ymin':[ymin], 'dw':[xmax-xmin], 'dh':[ymax-ymin]}
    dat['xmax'] = [xmax]
    dat['ymax'] = [ymax]
    
    ds = ColumnDataSource(data=dat)
    
    plot = figure(x_range=[xmin,xmax], y_range=[ymin,ymax], 
                  x_axis_label = xlabel,  y_axis_label = ylabel,
               plot_width=500, plot_height=500)
    plot.image(image='image', x='xmin', y='ymin', dw='dw', dh='dh', source=ds,palette=pal)
    
    return plot


def plot_histogram_h5(h5_bunch, component1, bins=30, nice=True, liveOnly=True):
    #c_light = 299792458. 
    #total_charge = h5_bunch.attrs['totalCharge']
    
    
    
    dat = particle_array(h5_bunch, component1, liveOnly=liveOnly)
    weights = particle_array(h5_bunch, 'weight', liveOnly=True)
    if len(weights)==1:
        weights = None
    
    hist, edges = np.histogram(particle_array(h5_bunch, component1, liveOnly=liveOnly), 
                               weights = weights,
                               density=True, bins=bins)
    
    #dx = edges[1]-edges[0]
    
    if nice:
        f1 = nice_phase_space_factor[component1]
        edges *= f1
        xlabel=nice_phase_space_label[component1]
        ylabel =  '??particles/'+nice_phase_space_unit[component1]
    else:
        xlabel = component1
        ylabel = '??particles/bin'
    
    
    # Change units
    #hist *= total_charge * c_light / 1000 # q/m * c  = q/s = A. Then change from A = 1/1000 kA
    #edges *=  1e15/c_light
    
    plot = figure(plot_width=500, plot_height=250, x_axis_label = xlabel, y_axis_label=ylabel)
    plot.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:],
       fill_color='grey', line_color="white", alpha=0.5)
    return plot


def plot_bunch_current_profile(h5_bunch, bins=30, liveOnly=True):
    c_light = 299792458. 
    total_charge = h5_bunch.attrs['chargeLive']
    
    weights = particle_array(h5_bunch, 'weight', liveOnly=liveOnly)
    if len(weights)==1:
        weights = None
    
    hist, edges = np.histogram(particle_array(h5_bunch, 'z', liveOnly=liveOnly),
                               density=True, weights = weights, bins=bins)
    
    # Change units
    hist *= total_charge * c_light / 1000 # q/m * c  = q/s = A. Then change from A = 1/1000 kA
    edges *=  1e15/c_light
    
    plot = figure(plot_width=500, plot_height=250, x_axis_label = 'z/c (fs)', y_axis_label='current (kA)')
    plot.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:],
       fill_color='grey', line_color="white", alpha=0.5)
    return plot



def bin_bunch_datasource_h5(h5_bunch, component1, component2, bins=20, nice=True, liveOnly=True):
    H, xedges, yedges = bin_particles2d_h5(h5_bunch, component1, component2, bins, liveOnly=liveOnly)
    xmin = min(xedges)
    xmax = max(xedges)
    ymin = min(yedges)
    ymax = max(yedges)
    
    
    if nice:
        f1 = nice_phase_space_factor[component1]
        f2 = nice_phase_space_factor[component2]
        xlabel =  nice_phase_space_label[component1]
        ylabel =  nice_phase_space_label[component2]
        xmin *= f1
        xmax *= f1
        ymin *= f2
        ymax *= f2
    else:
        xlabel = component1
        ylabel = component2
    
    # Form datasource
    dat = {'image':[H.transpose()], 'xmin':[xmin], 'ymin':[ymin], 'dw':[xmax-xmin], 'dh':[ymax-ymin]}
    dat['xmax'] = [xmax]
    dat['ymax'] = [ymax]
    
    ds = ColumnDataSource(data=dat)
    
    return ds



def plot_bunch_grid_h5(h5_bunch, bins=100, plot_width=200, plot_height=200 ):
    

    allplots = []
    components = [ 'z', 'pz', 'x', 'px', 'y', 'py']
    for c1, c2  in itertools.combinations(components, 2):
        p = plot_bunch_h5(h5_bunch, c1, c2, bins=bins)
        allplots.append(p)
    mplots = [allplots[0:5],allplots[5:10], allplots[10:15]]    

    grid = gridplot(mplots,  plot_width=plot_width, plot_height=plot_height)
    #, sizing_mode='fixed', toolbar_location='above', ncols=None, plot_width=None, plot_height=None, toolbar_options=None, merge_tools=True)
    return grid




def plot_slice_statistics(slice_stats, component1, bins=30, nice=True):
    #c_light = 299792458. 
    #total_charge = h5_bunch.attrs['totalCharge']
    
    hist, edges = np.histogram(particle_array(h5_bunch, component1), density=True, bins=bins)
    
    if nice:
        f1 = nice_phase_space_factor[component1]
        edges *= f1
        xlabel=nice_phase_space_label[component1]
        ylabel =  '??particles/'+nice_phase_space_unit[component1]
    else:
        xlabel = component1
        ylabel = '??particles/bin'
    
    
    # Change units
    #hist *= total_charge * c_light / 1000 # q/m * c  = q/s = A. Then change from A = 1/1000 kA
    #edges *=  1e15/c_light
    
    plot = figure(plot_width=500, plot_height=250, x_axis_label = xlabel, y_axis_label=ylabel)
    plot.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:],
       fill_color='grey', line_color="white", alpha=0.5)
    return plot
