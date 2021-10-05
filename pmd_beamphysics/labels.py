


TEXLABEL = {
# 'status'
 't': 't',
 'energy': 'E',
 'kinetic_energy': r'E_{kinetic}',
# 'mass',
# 'higher_order_energy_spread',
# 'higher_order_energy',
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
# 'species_charge',
# 'weight',
 #'average_current',
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
        any pmd_beamphysics attribure
    
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
    
    
    # Not found
    #raise ValueError(f'Unable to form tex label for {key}')
    
    return key
    
