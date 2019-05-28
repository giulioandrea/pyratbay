# Copyright (c) 2016-2019 Patricio Cubillos and contributors.
# Pyrat Bay is currently proprietary software (see LICENSE).

__all__ = ['CCSgray', 'Deck']

import numpy as np

from ... import constants as pc
from ... import tools     as pt


class CCSgray():
  """
  Constant cross-section gray cloud model.
  """
  def __init__(self):
      self.name  = 'ccsgray'       # Model name is lowercased class name
      self.pars  = [0.0,           # log10 of cross-section scale factor, top,
                   -4.0, 2.0]      #  and bottom pressure (bar) boundaries
      self.npars = len(self.pars)  # Number of model fitting parameters
      self.ec    = None            # Model extinction coefficient
      self.mol   = 'H2'            # Species causing the extinction
      # Fitting-parameter names (plain text and figure labels):
      self.pnames = ['log(f_gray)', 'log(p_top)', 'log(p_bot)']
      self.texnames = [r'$\log_{10}(f_{\rm gray})$',
                       r'$\log_{10}(p_{\rm top})\ ({\rm bar})$',
                       r'$\log_{10}(p_{\rm bot})\ ({\rm bar})$']
      self.s0 = 5.31e-27         # Default coss-section (cm-2 molec-1)

  def extinction(self, wn, pressure):
      """
      Calculate a uniform gray-cloud cross section in cm2 molec-1:
         cross section = s0 * 10**pars[0],
      between layers with pressure 10**pars[1] -- 10**pars[2] bar
      (top and bottom layers, respectively).
      s0 is the H2 Rayleigh cross section at 0.35 um.

      Parameters
      ----------
      wn:  1D float ndarray
         Wavenumber array in cm-1.
      """
      nlayers = len(pressure)
      nwave   = len(wn)
      # Get indices for cloud layer boundaries:
      itop    = np.where(pressure >= 10**self.pars[1]*pc.bar)[0][ 0]
      ibottom = np.where(pressure <= 10**self.pars[2]*pc.bar)[0][-1]
      # Gray opacity cross section in cm2 molec-1 (aka. extinction coef.):
      self.ec = np.zeros((nlayers, nwave))
      self.ec[itop:ibottom+1,:] = 10**self.pars[0] * self.s0

  def __str__(self):
      fw = pt.Formatted_Write()
      fw.write("Model name (name): '{}'", self.name)
      fw.write('Model species (mol): {}', self.mol)
      fw.write('Number of model parameters (npars): {}', self.npars)
      fw.write('Parameter name     Value\n'
               '  (pnames)         (pars)\n')
      for pname, param in zip(self.pnames, self.pars):
          fw.write('  {:15s}  {: .3e}', pname, param)
      fw.write('Extinction-coefficient (ec, cm2 molec-1):\n{}', self.ec,
          fmt={'float':'{: .3e}'.format}, edge=3)
      return fw.text


class Deck():
  """
  Instantly opaque gray cloud deck at given pressure.
  """
  def __init__(self):
      self.name  = 'deck'          # Model name is lowercased class name
      self.pars  = [-1.0]          # log10(Pressure[bar]) of cloud top
      self.npars = len(self.pars)  # Number of model fitting parameters
      self.ec    = None            # Model extinction coefficient
      # Fitting-parameter names (plain text and figure labels):
      self.pnames = ['log(p_top)']
      self.texnames = [r'$\log_{10}(p_{\rm top})$']

  def extinction(self, wn, pressure, radius):
      """
      Calculate gray-cloud absorption (in cm-1) that's optically thin
      above ptop, and becomes nearly-instantly opaque at ptop, with
      ptop (bar) = 10**pars[0].

      Parameters
      ----------
      wn: 1D float ndarray
          Wavenumber array (in cm-1).
      pressure: 1D float ndarray
          Atmospheric pressure profile (in barye).
      radius: 1D float ndarray
          Atmospheric radius profile (in cm).
      """
      nlayers = len(pressure)
      nwave   = len(wn)
      ptop = 10**self.pars[0]*pc.bar
      # Index of layer directly below cloud top:
      if ptop >= pressure[-1]:  # Atmosphere boundary cases
          itop = nlayers-1
      elif ptop < pressure[0]:
          itop = 1
      else:
          itop = np.where(pressure>ptop)[0][0]

      # Set the extinction at itop such that vertical tau[itop] ~ 2/3:
      alpha = (2.0/3)/ptop
      ec = np.zeros(nlayers)
      ec[itop] = 2*alpha*(pressure[itop])/(radius[itop-1]-radius[itop])
      # Below itop, scale the extinction proportional to the pressure:
      ec[itop:] = ec[itop] * pressure[itop:]/pressure[itop]

      # Gray absorption in cm-1:
      self.ec = np.zeros((nlayers, nwave))
      self.ec[itop:,:] = np.expand_dims(ec[itop:], axis=1)

  def __str__(self):
      fw = pt.Formatted_Write()
      fw.write("Model name (name): '{}'", self.name)
      fw.write('Number of model parameters (npars): {}', self.npars)
      fw.write('Parameter name     Value\n'
               '  (pnames)         (pars)\n')
      for pname, param in zip(self.pnames, self.pars):
          fw.write('  {:15s}  {: .3e}', pname, param)
      fw.write('Extinction-coefficient (ec, cm-1):\n{}', self.ec,
          fmt={'float':'{: .3e}'.format}, edge=3)
      return fw.text
