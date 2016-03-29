import scipy.constants as sc

"""
Constant values used in the pyrat project.

Notes
-----
  Solar system constants come from:
  http://nssdc.gsfc.nasa.gov/planetary/factsheet/
"""

# Universal constants:
h = sc.h * 1e7  # Planck constant
k = sc.k * 1e7  # Boltzmann constant in ergs/kelvin
c = sc.c * 1e2  # Speed of light in cm/s

# Convert from eV to cm-1 (kayser):
# planck   = 6.62620e-34  # Planck constant [J * s]
# lumiere  = 2.997925e10  # speed of light  [cm / s]
# electron = 1.602192e-19 # elementary charge [Coulomb]
# kayser2eV = planck * lumiere / electron
eV = 8065.49179

# Distance to cm:
A  = 1e-8  # Angstrom
nm = 1e-7  # Nanometer
um = 1e-4  # Microns
mm = 1e-1  # Millimeter
cm = 1.0   # Centimeter
m  = 1e+2  # Meter
km = 1e+5  # Kilometer
au = sc.au*100      # Astronomical unit
pc = sc.parsec*100  # Parsec
rearth = 6.3710e8  # Earth radius
rjup   = 6.9911e9  # Jupiter mean radius
rsun   = 6.955e10  # Sun radius

# Pressure to Barye:
barye  = 1.0    # Barye (CGS units)
mbar   = 1e3    # Millibar
pascal = 1e5    # Pascal (MKS units)
bar    = 1e6    # Bar
atm    = 1.01e6 # Atmosphere

# Mass to grams:
mearth = 5.9724e27     # Earth mass
mjup   = 1.8982e30     # Jupiter mass
msun   = 1.9885e33     # Sun mass
# Unified atomic mass:
amu    = sc.physical_constants["unified atomic mass unit"][0] * 1e3
me     = sc.m_e * 1e3  # Electron mass

# Temperature to Kelvin degree:
kelvin = 1.0

# Amagat (Loschmidt number) molecules cm-3:
amagat = sc.physical_constants[
                 "Loschmidt constant (273.15 K, 101.325 kPa)"][0] * 1e-6

# Elementary charge in statcoulombs (from Wolfram Alpha):
e = 4.803205e-10

# No units:
none = 1

# Valid units for conversion:
validunits = ["A", "nm", "um", "mm", "cm", "m", "km", "au", "pc",
              "rearth", "rjup", "rsun",
              "barye", "mbar", "pascal", "bar", "atm",
              "kelvin",
              "eV",
              "amu", "me", "mearth", "mjup", "msun",
              "amagat", "none"]

# Other combination of constants:
C1 = 4 * sc.epsilon_0 * sc.m_e * sc.c**2 / sc.e**2 * 0.01  # cm-1
C2 = sc.h * (sc.c * 100.0) / sc.k                          # cm / Kelvin
C3 = sc.pi * e**2 / (me * c**2)                            # cm

# String lengths:
maxnamelen = 20
strfmt = "|S%d"%maxnamelen

# TLI record lengths:
tlireclen = 26  # Three doubles and one short
dreclen   =  8  # Double  byte length
ireclen   =  4  # Integer byte length
sreclen   =  2  # Short   byte length
