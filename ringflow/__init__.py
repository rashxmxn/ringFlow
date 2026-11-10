"""ringflow -- Python port of the 2015 Fortran programs for numerical
modelling of free convection in ring-shaped (doubly-connected) domains.

Modules
-------
boundary    port of physical_area/interp.for      (boundary point placement)
mesh        port of Dvusv_set_dvish/Setka_ED2_II  (elliptic mesh + quality)
convection  port of Dvizh_zhid/Dvizh_psi_om_Kolc*.for (psi-omega-T solver)
io          Tecplot / xxx.dat readers & writers, VTK export
plotting    pictures and animations (new)
cli         command line interface:  python -m ringflow --help
"""

__version__ = "1.0.0"

from . import boundary, mesh, convection, io  # noqa: F401
