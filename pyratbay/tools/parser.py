# Copyright (c) 2016-2019 Patricio Cubillos and contributors.
# Pyrat Bay is currently proprietary software (see LICENSE).

__all__ = ['Namespace',
           'parse',
           'parse_str', 'parse_int', 'parse_float', 'parse_array']

import os
import sys
import argparse
from datetime import date
if sys.version_info.major == 3:
    import configparser
else:
    import ConfigParser as configparser

import numpy as np

from .  import tools     as pt
from .. import VERSION   as ver
from .. import constants as pc

sys.path.append(pc.ROOT + "modules/MCcubed")
import MCcubed.utils as mu


class Namespace(argparse.Namespace):
    """A container object to hold variables."""
    def __init__(self, args=None, log=None):
        if args is not None:
            super(Namespace, self).__init__(**args)
        if log is None:
            self._log = mu.Log(logname=None, verb=2, width=80)
        else:
            self._log = log

    def get_path(self, pname, desc='', exists=False):
        """
        Extract pname file path (or list of paths) from Namespace,
        return the canonical path.

        Examples
        --------
        >>> import pyratbay.tools as pt
        >>> ns = pt.Namespace({'f1':'file1', 'f23':['file2', 'file3']})
        >>> # Get path of a single file:
        >>> ns.get_path('f1')
        >>> # Get path of a list of files:
        >>> ns.get_path('f23')
        >>> # Attempt to get non-existing file:
        >>> ns.get_path('f1', desc='Configuration', exists=True)
        """
        value = getattr(self, pname)
        if value is None:
            return None

        if isinstance(value, str):
            values = [value]
            is_list = False
        else:
            values = value
            is_list = True

        for val in values:
            if exists and not os.path.isfile(val):
                self._log.error("{} file ({}) does not exist: '{}'".
                                format(desc, pname, val))

        values = [os.path.realpath(val) for val in values]
        for val in values:
            if not os.path.exists(os.path.dirname(val)):
                self._log.error("Dir for {} file ({}) does not exist: '{}'".
                                format(desc, pname, val))

        if not is_list:
            return values[0]
        return values

    def get_choice(self, pname, desc, choices, take_none=True):
        value = getattr(self, pname)
        if value is None and take_none:
            return None

        if isinstance(value, list):
            values = value
            is_list = True
        else:
            values = [value]
            is_list = False

        for value in values:
            if value not in choices:
                self._log.error("Invalid {:s} ({:s}): {}. Select from: {:s}.".
                    format(desc, pname, value, str(choices)), tracklev=-3)
        if not is_list:
            return values[0]
        return values

    def get_default(self, pname, desc, default=None, wflag=False,
                    gt=None, ge=None, lt=None, le=None):
        """
        Extract pname variable from Namespace; if None, return
        default.  If any of gt, ge, lt, or le is not None, run
        greater/lower/equal checks.

        Parameters
        ----------
        pname: String
            Parameter name.
        desc: String
            Parameter description.
        default: Any
            Parameter default value.
        gt: Float
            If not None, check output is greater than gt.
        ge: Float
            If not None, check output is greater-equal than gt.
        lt: Float
            If not None, check output is lower than gt.
        le: Float
            If not None, check output is lower-equal than gt.
        """
        value = getattr(self, pname)
        if value is None and default is not None:
            if wflag:
                self._log.warning('{} ({}) defaulted to: {}'.
                                  format(desc, pname, default))
            value = default

        if value is None:
            return None

        if gt is not None and value <= gt:
            self._log.error('{} ({}) must be > {}'.format(desc, pname, gt),
                            tracklev=-3)
        if ge is not None and value < ge:
            self._log.error('{} ({}) must be >= {}'.format(desc, pname, ge),
                            tracklev=-3)
        if lt is not None and lt <= value:
            self._log.error('{} ({}) must be < {}'.format(desc, pname, lt),
                            tracklev=-3)
        if le is not None and le < value:
            self._log.error('{} ({}) must be <= {}'.format(desc, pname, le),
                            tracklev=-3)
        return value

    def get_param(self, pname, units, desc, gt=None, ge=None):
        value = pt.get_param(pname, getattr(self, pname), units, self._log,
            tracklev=-4)
        if value is None:
            return None

        if gt is not None and value <= gt:
            self._log.error('{} ({}) must be > {}'.format(desc, pname, gt),
                            tracklev=-3)
        if ge is not None and value < ge:
            self._log.error('{} ({}) must be >= {}'.format(desc, pname, ge),
                            tracklev=-3)
        return value


def parse_str(args, param):
    """Parse a string parameter into args."""
    if param not in args:
        args[param] = None
    else:
        args[param] = str(args[param])


def parse_int(args, param):
    """
    Convert a dictionary's parameter from string to integer.
    Raise ValueError if the operation is not possible.
    Set parameter to None if it was not in the dictinary.

    Parameters
    ----------
    args: dict
        Dictionary where to operate.
    param: String
        Parameter to cast to int.

    Examples
    --------
    >>> import pyratbay.tools as pt
    >>> inputs = ['10', '-10', '+10', '10.0', '1e1',
    >>>           '10.5', 'None', 'True', 'inf', '10 20']
    >>> args = {'par{}'.format(i):val for i,val in enumerate(inputs)}
    >>> for i,var in enumerate(inputs):
    >>>     try:
    >>>         par = 'par{}'.format(i)
    >>>         pt.parse_int(args, par)
    >>>         print("{:s}: '{:s}' -> {}".format(par, var, args[par]))
    >>>     except ValueError as e:
    >>>         print(e)
    par0: '10' -> 10
    par1: '-10' -> -10
    par2: '+10' -> 10
    par3: '10.0' -> 10
    par4: '1e1' -> 10
    Invalid data type for par5, could not convert string to integer: '10.5'
    Invalid data type for par6, could not convert string to integer: 'None'
    Invalid data type for par7, could not convert string to integer: 'True'
    Invalid data type for par8, could not convert string to integer: 'inf'
    Invalid data type for par9, could not convert string to integer: '10 20'
    """
    if param not in args:
        args[param] = None
        return

    try:
        val = np.double(args[param])
    except:
        raise ValueError("Invalid data type for {}, could not convert string "
                         "to integer: '{:s}'".format(param, args[param]))
    if not np.isfinite(val) or int(val) != val:
        raise ValueError("Invalid data type for {}, could not convert string "
                         "to integer: '{:s}'".format(param, args[param]))
    args[param] = int(val)



def parse_float(args, param):
    """
    Convert a dictionary's parameter from string to float.
    Raise ValueError if the operation is not possible.
    Set parameter to None if it was not in the dictinary.

    Parameters
    ----------
    args: dict
        Dictionary where to operate.
    param: String
        Parameter to cast to float.

    Examples
    --------
    >>> import pyratbay.tools as pt
    >>> inputs = ['10', '-10', '+10', '10.5', '1e1', 'inf', 'nan',
    >>>           'None', 'True', '10 20']
    >>> args = {'par{}'.format(i):val for i,val in enumerate(inputs)}
    >>> for i,var in enumerate(inputs):
    >>>     try:
    >>>         par = 'par{}'.format(i)
    >>>         pt.parse_float(args, par)
    >>>         print("{:s}: '{:s}' -> {}".format(par, var, args[par]))
    >>>     except ValueError as e:
    >>>         print(e)
    par0: '10' -> 10.0
    par1: '-10' -> -10.0
    par2: '+10' -> 10.0
    par3: '10.5' -> 10.5
    par4: '1e5' -> 10.0
    par5: 'inf' -> inf
    par6: 'nan' -> nan
    Invalid data type for par7, could not convert string to float: 'None'
    Invalid data type for par8, could not convert string to float: 'True'
    Invalid data type for par9, could not convert string to float: '10 20'
    """
    if param not in args:
        args[param] = None
        return

    try:
        val = np.double(args[param])
    except:
        raise ValueError("Invalid data type for {}, could not convert string "
                         "to float: '{:s}'".format(param, args[param]))
    args[param] = val


def parse_array(args, param):
    r"""
    Convert a dictionary's parameter from string to iterable.
    If possible cast into a float numpy array; otherwise,
    set as a list of strings.
    Assume any blank character delimits the elements in the string.
    Set parameter to None if it was not in the dictinary.

    Parameters
    ----------
    args: dict
        Dictionary where to operate.
    param: String
        Parameter to cast to array.

    Examples
    --------
    >>> import pyratbay.tools as pt
    >>> inputs = ['10 20', '10.0 20.0', 'a b', 'a\n b']
    >>> args = {'par{}'.format(i):val for i,val in enumerate(inputs)}
    >>> for i,var in enumerate(inputs):
    >>>     par = 'par{}'.format(i)
    >>>     pt.parse_array(args, par)
    >>>     print("{:s}: {:s} -> {}".format(par, repr(var), repr(args[par])))
    par0: '10 20' -> array([10., 20.])
    par1: '10.0 20.0' -> array([10., 20.])
    par2: 'a b' -> ['a', 'b']
    par3: 'a\n b' -> ['a', 'b']
    """
    if param not in args:
        args[param] = None
        return

    val = args[param].split()
    try:
        val = np.asarray(val, np.double)
    except:
        pass
    args[param] = val


def parse(pyrat, cfile):
  """
  Read the command line arguments.

  Parameters
  ----------
  cfile: String
      A Pyrat Bay configuration file.

  Returns
  -------
  args: Namespace
      Object storing the attributes defined in this function, with
      the values given in cfile.
  log: Log object
      An MCcubed.utils.Log instance to log screen outputs to file.
  """
  if cfile is None:
      print("No configuration file specified.")
      sys.exit(0)
  elif not os.path.isfile(cfile):
      print("Configuration file '{:s}' does not exist.".format(cfile))
      sys.exit(0)

  config = configparser.ConfigParser()
  config.optionxform = str  # Enable case-sensitive variable names
  config.read([cfile])
  if "pyrat" not in config.sections():
      print("\nInvalid configuration file: '{:s}', no [pyrat] section.".
            format(cfile))
      sys.exit(0)
  args = dict(config.items("pyrat"))

  # Parse data type:
  with pt.log_error():
      parse_int(args,   'verb')
      parse_array(args, 'dblist')
      parse_array(args, 'pflist')
      parse_array(args, 'dbtype')
      parse_array(args, 'tlifile')
      parse_array(args, 'csfile')
      parse_str(args,   'molfile')
      parse_str(args,   'extfile')
      # Spectrum sampling options:
      parse_str(args,   'wlunits')
      parse_str(args,   'wllow')
      parse_str(args,   'wlhigh')
      parse_float(args, 'wnlow')
      parse_float(args, 'wnhigh')
      parse_float(args, 'wnstep')
      parse_int(args,   'wnosamp')
      parse_float(args, 'resolution')
      # Atmospheric sampling options:
      parse_str(args,   'tmodel')
      parse_array(args, 'tpars')
      parse_str(args,   'radlow')
      parse_str(args,   'radhigh')
      parse_str(args,   'radstep')
      parse_str(args,   'runits')
      parse_str(args,   'punits')
      parse_int(args,   'nlayers')
      parse_str(args,   'ptop')
      parse_str(args,   'pbottom')
      parse_str(args,   'atmfile')
      # Variables for TEA calculations
      parse_array(args, 'species')
      parse_array(args, 'uniform')
      parse_str(args,   'ptfile')
      parse_str(args,   'solar')
      parse_float(args, 'xsolar')
      parse_str(args,   'atomicfile')
      parse_str(args,   'patm')
      parse_array(args, 'elements')
      # Extinction options:
      parse_float(args, 'tmin')
      parse_float(args, 'tmax')
      parse_float(args, 'tstep')
      parse_float(args, 'ethresh')
      parse_int(args,   'ncpu')
      # Voigt-profile options:
      parse_float(args, 'vextent')
      parse_float(args, 'Dmin')
      parse_float(args, 'Dmax')
      parse_int(args,   'nDop')
      parse_float(args, 'Lmin')
      parse_float(args, 'Lmax')
      parse_int(args,   'nLor')
      parse_float(args, 'DLratio')
      # Hazes and clouds options:
      parse_array(args, 'hazes')
      parse_array(args, 'hpars')
      parse_array(args, 'rayleigh')
      parse_array(args, 'rpars')
      parse_float(args, 'fpatchy')
      parse_array(args, 'alkali')
      # Optical depth options:
      parse_str(args,   'path')
      parse_float(args, 'maxdepth')
      parse_array(args, 'raygrid')
      parse_int(args,   'quadrature')
      parse_str(args,   'runmode')
      # Data options:
      parse_array(args, 'data')
      parse_array(args, 'uncert')
      parse_array(args, 'filter')
      # Retrieval options:
      parse_array(args, 'retflag')
      parse_array(args, 'bulk')
      parse_array(args, 'molmodel')
      parse_array(args, 'molfree')
      parse_array(args, 'molpars')
      parse_float(args, 'qcap')
      parse_array(args, 'params')
      parse_array(args, 'stepsize')
      parse_float(args, 'tlow')
      parse_float(args, 'thigh')
      parse_str(args,   'mcmcfile')
      parse_array(args, 'pmin')
      parse_array(args, 'pmax')
      parse_array(args, 'prior')
      parse_array(args, 'priorlow')
      parse_array(args, 'priorup')
      parse_str(args,   'walk')        # 'snooker'
      parse_int(args,   'nsamples')    # 1e5
      parse_int(args,   'nchains')     # 7
      parse_int(args,   'burnin')      # 0
      parse_int(args,   'thinning')
      parse_float(args, 'grbreak')
      parse_float(args, 'grnmin')
      parse_int(args,   'resume')      # False, action='store_true')
      parse_str(args,   'starspec')
      parse_str(args,   'kurucz')
      parse_str(args,   'marcs')
      parse_str(args,   'phoenix')
      # System parameters:
      parse_str(args,   'rstar')
      parse_float(args, 'gstar')
      parse_float(args, 'tstar')
      parse_str(args,   'mstar')
      parse_str(args,   'rplanet')
      parse_str(args,   'refpressure')
      parse_str(args,   'mplanet')
      parse_float(args, 'gplanet')
      parse_str(args,   'smaxis')
      parse_float(args, 'tint')
      # Outputs:
      parse_str(args,   'outspec')
      parse_str(args,   'logfile')
      parse_array(args, 'logxticks')
      parse_array(args, 'yran')

  # Cast into a Namespace to make my life easier:
  args = Namespace(args)
  args.configfile = cfile

  pyrat.verb = args.get_default('verb', 'Verbosity', 2, ge=0, lt=5)
  runmode = pyrat.runmode = args.get_choice('runmode', 'running mode',
      pc.rmodes, take_none=False)

  # Define logfile name and initialize log object:
  pyrat.lt.tlifile   = args.get_path('tlifile',  'TLI')
  pyrat.atm.atmfile  = args.get_path('atmfile',  'Atmospheric')
  pyrat.spec.outspec = args.get_path('outspec',  'Output spectrum')
  pyrat.ex.extfile   = args.get_path('extfile',  'Extinction-coefficient')
  pyrat.ret.mcmcfile = args.get_path('mcmcfile', 'MCMC')

  if args.logfile is None:
      if args.runmode == 'tli' and args.tlifile is not None:
          args.logfile = os.path.splitext(args.tlifile[0])[0] + '.log'
      if args.runmode == 'atmosphere' and args.atmfile is not None:
          args.logfile = os.path.splitext(args.atmfile)[0] + '.log'
      if args.runmode == 'spectrum' and args.outspec is not None:
          args.logfile = os.path.splitext(args.outspec)[0] + '.log'
      if args.runmode == 'opacity' and args.extfile is not None:
          args.logfile = os.path.splitext(args.extfile)[0] + '.log'
      if args.runmode == 'mcmc' and args.mcmcfile is not None:
          args.logfile = os.path.splitext(args.mcmcfile)[0] + '.log'

  args.logfile = pt.path(args.logfile)
  log = pyrat.log = mu.Log(logname=args.logfile, verb=args.verb, width=80,
                           append=args.resume)
  args._log = log

  # Welcome message:
  log.msg("{:s}\n"
          "  Python Radiative Transfer in a Bayesian framework (Pyrat Bay).\n"
          "  Version {:d}.{:d}.{:d}.\n"
          "  Copyright (c) 2016-{:d} Patricio Cubillos and collaborators.\n"
          "  Pyrat Bay is (temporarily) proprietaty software (see LICENSE).\n"
          "{:s}\n\n".format(log.sep, ver.PBAY_VER, ver.PBAY_MIN,
                            ver.PBAY_REV, date.today().year, log.sep), verb=0)

  log.msg("Read command-line arguments from configuration file: '{:s}'".
          format(cfile))

  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # Parse valid inputs and defaults:
  pyrat.inputs = args

  phy  = pyrat.phy
  spec = pyrat.spec
  atm  = pyrat.atm

  pyrat.mol.molfile = args.get_path('molfile', 'Molecular data')
  pyrat.cs.files    = args.get_path('csfile',  'Cross-section')
  pyrat.atm.ptfile  = args.get_path('ptfile',  'Pressure-temperature')

  spec.wlunits = args.get_default('wlunits', 'Wavelength units', 'um',
      wflag=runmode not in ['pt', 'atmosphere'])
  spec.wllow  = args.get_param('wllow',  spec.wlunits,
      'Wavelength lower boundary',  gt=0.0)
  spec.wlhigh = args.get_param('wlhigh', spec.wlunits,
      'Wavelength higher boundary', gt=0.0)

  spec.wnlow  = args.get_default('wnlow',
      'Wavenumber lower boundary',  gt=0.0)
  spec.wnhigh = args.get_default('wnlow',
      'Wavenumber higher boundary', gt=spec.wnlow)

  spec.wnstep = args.get_default('wnstep',
      'Wavenumber sampling step', gt=0.0)
  spec.wnosamp = args.get_default('wnosamp',
      'Wavenumber oversampling factor', ge=1)
  spec.resolution = args.get_default('resolution',
      'Spectral resolution', gt=0.0)

  atm.punits = args.get_default('punits', 'Pressure units', 'bar',
      wflag=(runmode!='tli'))
  atm.runits = args.get_default('runits', 'Distance units', 'km',
      wflag=(runmode!='tli'))
  atm.nlayers = args.get_default('nlayers',
      'Number of atmospheric layers', gt=0)

  # Pressure boundaries:
  atm.pbottom = args.get_param('pbottom', atm.punits,
      'Pressure at bottom of atmosphere', gt=0.0)
  atm.ptop = args.get_param('ptop', atm.punits,
      'Pressure at top of atmosphere', gt=0.0)

  # Radius boundaries:
  atm.radlow = args.get_param('radlow', atm.runits,
      'Radius at bottom of atmosphere', ge=0.0)
  atm.radhigh = args.get_param('radhigh', atm.runits,
      'Radius at top of atmosphere', gt=0.0)
  atm.radstep = args.get_param('radstep', atm.runits,
      'Radius sampling step', gt=0.0)

  # System physical parameters:
  atm.refpressure = args.get_param('refpressure', atm.punits,
      'Planetary reference pressure level', gt=0.0)
  phy.rplanet = args.get_param('rplanet', atm.runits,
      'Planetary radius', gt=0.0)
  phy.mplanet = args.get_param('mplanet', None,
      'Planetary mass', gt=0.0)
  phy.gplanet = args.get_default('gplanet',
      'Planetary surface gravity (cm s-2)', gt=0.0)
  phy.tint = args.get_default('tint',
      'Planetary internal temperature', 100.0, gt=0)

  phy.smaxis = args.get_param('smaxis', atm.runits,
      'Orbital semi-major axis', gt=0.0)
  phy.rstar = args.get_param('rstar', atm.runits,
      'Stellar radius', gt=0.0)
  phy.mstar = args.get_param('mstar', 'msun',
      'Stellar mass', gt=0.0)
  phy.gstar = args.get_default('gstar',
      'Stellar surface gravity', gt=0.0)
  phy.tstar = args.get_default('tstar',
      'Stellar effective temperature (K)', gt=0.0)

  pyrat.voigt.extent = args.get_default('vextent',
      'Voigt profile extent', 20.0, ge=1.0,
      wflag=(runmode not in ['tli', 'pt', 'atmosphere']))
  pyrat.voigt.nDop = args.get_default('nDop',
      'Number of Doppler-width samples', 40, ge=1)
  pyrat.voigt.Dmin = args.get_default('Dmin',
      'Minimum Doppler HWHM (cm-1)', gt=0.0)
  pyrat.voigt.Dmax = args.get_default('Dmax',
      'Maximum Doppler HWHM (cm-1)', gt=0.0)
  pyrat.voigt.nLor = args.get_default('nLor',
      'Number of Lorentz-width samples', 40, ge=1)
  pyrat.voigt.Lmin = args.get_default('Lmin',
      'Minimum Lorentz HWHM (cm-1)', gt=0.0)
  pyrat.voigt.Lmax = args.get_default('Lmax',
      'Maximum Lorentz HWHM (cm-1)', gt=0.0)
  pyrat.voigt.DLratio = args.get_default('DLratio',
      'Doppler/Lorentz-width ratio threshold', 0.1, gt=0)

  pyrat.ex.tmin = args.get_param('tmin', 'kelvin',
      'Minimum temperature of opacity grid', gt=0.0)
  pyrat.ex.tmax = args.get_param('tmax', 'kelvin',
      'Maximum temperature of opacity grid', gt=pyrat.ex.tmin)
  pyrat.ex.tstep = args.get_default('tstep',
      "Opacity grid's temperature sampling step in K", gt=0.0)

  pyrat.rayleigh.pars = args.rpars
  pyrat.haze.pars     = args.hpars
  pyrat.rayleigh.model_names = args.get_choice('rayleigh',
      'Rayleigh model', pc.rmodels)
  pyrat.haze.model_names = args.get_choice('hazes',
      'aerosol model', pc.cmodels)
  pyrat.alkali.model_names = args.get_choice('alkali',
      'alkali model', pc.amodels)
  pyrat.haze.fpatchy = args.get_default('fpatchy',
      'Patchy-cloud fraction', ge=0.0, le=1.0)

  pyrat.od.path = args.get_choice('path',
      'observing geometry', ['transit','eclipse'])
  pyrat.ex.ethresh = args.get_default('ethresh',
      'Extinction-cofficient threshold', 1e-15, gt=0.0)
  pyrat.od.maxdepth = args.get_default('maxdepth',
      'Maximum optical-depth', 10.0, ge=0.0)

  phy.starspec = args.get_path('starspec', 'Stellar spectrum', exists=True)
  phy.kurucz   = args.get_path('kurucz',   'Kurucz model',     exists=True)
  phy.marcs    = args.get_path('marcs',    'MARCS model',      exists=True)
  phy.phoenix  = args.get_path('phoenix',  'PHOENIX model',    exists=True)

  spec.raygrid = args.get_default('raygrid',
      'Emission raygrid (deg)', np.array([0, 20, 40, 60, 80.]),
      wflag=(runmode not in ['tli', 'pt', 'atmosphere']))
  spec.quadrature = args.get_default('quadrature',
      'Number of Gaussian-quadrature points', ge=1)

  pyrat.obs.data   = args.data
  pyrat.obs.uncert = args.uncert
  pyrat.obs.filter = args.filter

  pyrat.ret.retflag = args.retflag

  pyrat.ret.params   = args.params
  pyrat.ret.stepsize = args.stepsize
  pyrat.ret.pmin     = args.pmin
  pyrat.ret.pmax     = args.pmax
  pyrat.ret.prior    = args.prior
  pyrat.ret.priorlow = args.priorlow
  pyrat.ret.priorup  = args.priorup

  pyrat.ret.qcap = args.get_default('qcap',
      'Metals abundance cap', 1.0, gt=0, le=1.0)
  pyrat.ret.params = args.params
  pyrat.ret.stepsize = args.stepsize
  pyrat.ret.tlow  = args.get_default('tlow',
      'Retrieval low-temperature (K) bound', 0, wflag=(runmode=='mcmc'))
  pyrat.ret.thigh = args.get_default('thigh',
      'Retrieval high-temperature (K) bound', np.inf, wflag=(runmode=='mcmc'))
  pyrat.ret.walk     = args.walk
  pyrat.ret.nsamples = args.get_default('nsamples',
      'Number of MCMC samples', gt=0)
  pyrat.ret.burnin   = args.get_default('burnin',
      'Number of burn-in samples per chain', gt=0)
  pyrat.ret.thinning = args.get_default('thinning',
      'MCMC posterior thinning', 1)
  pyrat.ret.nchains  = args.get_default('nchains',
      'Number of MCMC parallel chains', ge=1)
  pyrat.ret.grbreak  = args.get_default('grbreak',
      'Gelman-Rubin convergence criteria', 0.0, ge=0)
  pyrat.ret.grnmin   = args.get_default('grnmin',
      'Gelman-Rubin convergence fraction', 0.5, gt=0.0)
  atm.molmodel = args.molmodel
  atm.molfree  = args.molfree
  atm.molpars  = args.molpars
  atm.bulk     = args.bulk
  atm.tmodelname = args.get_choice('tmodel', 'temperature model', pc.tmodels)
  atm.tpars = args.tpars
  pyrat.ncpu = args.get_default('ncpu', 'Number of processors', 1, ge=1)

  return