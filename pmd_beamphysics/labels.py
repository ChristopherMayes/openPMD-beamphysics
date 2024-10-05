from pmd_beamphysics.units import pg_units, parse_bunching_str, nice_array


TEXLABEL = {
# 'status'
 't': 't',
 'energy': 'E',
 'kinetic_energy': r'E_{kinetic}',
# 'mass',
# 'higher_order_energy_spread',
# 'higher_order_energy',
 'Ex': 'E_x',
 'Ey': 'E_y',
 'Ez': 'E_z',    
 'Bx': 'B_x',
 'By': 'B_y',
 'Bz': 'B_z',    
 'Etheta': r'E_{\theta}',
 'Btheta': r'B_{\theta}',    
 'px': 'p_x',
 'py': 'p_y',
 'pz': 'p_z',
 'p': 'p',
 'pr': 'p_r',
 'ptheta': r'p_{\theta}',
 'x': 'x',
 'y': 'y',
 'z': 'z',
 'r': 'r',
 'Jx': 'J_x',
 'Jy': 'J_y',
 'beta': r'\beta',
 'beta_x': r'\beta_x',
 'beta_y': r'\beta_y',
 'beta_z': r'\beta_z',
 'gamma': r'\gamma',
 'theta': r'\theta',
 'charge': 'Q',
 'twiss_alpha_x'     : r"Twiss\ \alpha_x",
 'twiss_beta_x'      : r"Twiss\ \beta_x",
 'twiss_gamma_x'     : r"Twiss\ \gamma_x",
 'twiss_eta_x'       : r"Twiss\ \eta_x",
 'twiss_etap_x'      : r"Twiss\ \eta'_x",     
 'twiss_emit_x'      : r"Twiss\ \epsilon_{x}",
 'twiss_norm_emit_x' : r"Twiss\ \epsilon_{n, x}",
 'twiss_alpha_y'     : r"Twiss\ \alpha_y",
 'twiss_beta_y'      : r"Twiss\ \beta_y",
 'twiss_gamma_y'     : r"Twiss\ \gamma_y",
 'twiss_eta_y'       : r"Twiss\ \eta_y",
 'twiss_etap_y'      : r"Twiss\ \eta'_y",     
 'twiss_emit_y'      : r"Twiss\ \epsilon_{y}",
 'twiss_norm_emit_y' : r"Twiss\ \epsilon_{n, y}",
# 'species_charge',
# 'weight',
 'average_current': r'I_{av}',
 'norm_emit_x': r'\epsilon_{n, x}',
 'norm_emit_y': r'\epsilon_{n, y}',
 'norm_emit_4d':  r'\epsilon_{4D}',
 'Lz': 'L_z',
 'xp': "x\'",
 'yp': "y\'",
 'x_bar': r'\overline{x}',
 'px_bar': r'\overline{p_x}',   
 'y_bar': r'\overline{y}',
 'py_bar': r'\overline{p_y}'        
}

def texlabel(key: str):
    """
    Returns a tex label from a proper attribute name.
    
    Parameters
    ----------
    key : str
        any pmd_beamphysics attribute
    
    Returns
    -------
    tex: str or None
        A TeX string if applicable, otherwise will return None
    
    
    Examples
    --------
        texlabel('cov_x__px')    
        returns: '\\left<x, p_x\\right>'
        
        
        
    Notes:
    -----
        See matplotlib: 
            https://matplotlib.org/stable/tutorials/text/mathtext.html
    
    """
    
    # Basic cases
    if key in TEXLABEL:
        return TEXLABEL[key]
    
    for prefix in ['sigma_', 'mean_', 'min_', 'max_', 'ptp_', 'delta_']:
        if key.startswith(prefix):
            pre = prefix[:-1]
            key0 = key[len(prefix):]
            tex0 = texlabel(key0)
            
            if pre in [ 'min', 'max']:
                return  f'\\{pre}({tex0})'
            if pre == 'sigma':
                return rf'\sigma_{{ {tex0} }}'
            if pre == 'delta':
                return fr'{tex0} - \left<{tex0}\right>'
            if pre == 'mean':
                return fr'\left<{tex0}\right>'
    
    if key.startswith('cov_'):
        subkeys = key.strip('cov_').split('__')
        tex0 = texlabel(subkeys[0])
        tex1 = texlabel(subkeys[1])
        return fr'\left<{tex0}, {tex1}\right>'
    
    if key.startswith('bunching'):
        wavelength = parse_bunching_str(key)
        x, _, prefix = nice_array(wavelength)
        return fr'\mathrm{{bunching~at}}~{x:.1f}~\mathrm{{ {prefix}m }}'
    
    return None
    



def mathlabel(*keys, units=None, tex=True):
    """
    Helper function to return label with optional units
    from an arbitrary number of keys
    

    
    Parameters
    ----------
    *keys : str
        any pmd_beamphysics attributes
        
    units : pmd_unit or str or None
        units to be cast to str. 
        
    tex : bool, default=True
        if True, a mathtext (TeX) string wrapped in $$ will be returned.
        Uses pmd_beamphysics.labels.texlabel to get a proper label
        
    Returns
    -------
    label: str
        A TeX string if applicable, otherwise will return None 
        
        
    Examples
    --------
        mathlabel('x_bar', 'sigma_x', units='µC')
        returns:
        '$\\overline{x}, \\sigma_{ x }~(\\mathrm{ µC } )$'
    
    """
    # Cast to str
    if units:
        units = str(units)

    if tex:
        l = [texlabel(key) or fr'\mathrm{{ {key} }}' for key in keys]
        label = ', '.join(l)
        if units:
            label = fr'{label}~(\mathrm{{ {units} }} )'
            
        return fr'${label}$'
        
    else:
        label = ', '.join(keys)
        
        if units:
            label = fr'{label} ({units})'
            
        return label
