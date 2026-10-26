# Plotting

Plots are rendered by a selectable backend: Matplotlib (`"mpl"`, the default)
or Bokeh (`"bokeh"`, requires the optional `bokeh` dependency). Pass
`backend=` to any plot method, set the `BEAMPHYSICS_PLOT` environment
variable, or change the default at runtime:

```python
import beamphysics

beamphysics.set_default_backend("bokeh")
```

::: beamphysics.set_default_backend

::: beamphysics.get_default_backend

::: beamphysics.plot_dispatch.get_backend
