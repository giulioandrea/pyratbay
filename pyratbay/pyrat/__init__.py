# Copyright (c) 2016-2020 Patricio Cubillos.
# Pyrat Bay is open-source software under the GNU GPL-2.0 license (see LICENSE).

__all__ = ['Pyrat']

from .pyrat_obj import Pyrat


# Clean up top-level namespace--delete everything that isn't in __all__
# or is a magic attribute, and that isn't a submodule of this package
for varname in dir():
    if not ((varname.startswith('__') and varname.endswith('__')) or
            varname in __all__ ):
        del locals()[varname]
del(varname)
