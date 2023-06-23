# Tools for dealing with the new PARSEC 2.0 isochrones

import numpy as np
from scipy import interpolate, stats
from astropy.table import Table

class Isochrones():
    """
    Class collecting some tools to deal with PARSEC isochrones.
    Initialise it by typing, e.g.,
        >>> iso = Isochrones("parsec20")
    """
    def __init__(self, version="parsec20", omega_i=0.3, phot="gaiaedr3"):
        """
        If present, read the isochrone file.
        """
        if version == "parsec20" and omega_i == 0.3 and phot == "gaiaedr3":
            self.data = Table.read("./data/isochrones/parsec_20_omegai03.dat", 
                                   format="ascii", header_start=14)
        else: 
            raise ValueError("please first download data and implement..")

    def select_age(self, log_age):
        """
        Finds and selects the nearest isochrone for a given age

        Args:
            log_age: float
        Returns: 
            isochrone astropy Table restricted to the given log_age
        """
        # Find the nearest bin to the input log_age
        age_bins = np.unique(self.data["logAge"])
        age_bin  = min(age_bins, key=lambda x:abs(x-log_age))
        # Cut out this bin
        return self.data[ self.data["logAge"] == age_bin ]

    def get_G_mass_relation(self, log_age, plot=False):
        """
        Interpolates the mass - Gmag relation for a given age

        Args:
            log_age: float
        Returns: 
            f_mass:  Interpolator function
        """
        # Restrict the models to the given age
        isoc_age = self.select_age(log_age)
        # Interpolate the Gmag - mass relation
        f_mass = interpolate.interp1d(isoc_age['G_fSBmag'], isoc_age['Mass'], 
                                      fill_value="extrapolate", bounds_error=False)
        # If wanted, plot the relationship
        if plot:
            x = np.linspace(-0.35, 10, 1000)
            plt.scatter(mag, mass, s=5, color ="yellow")
            plt.plot(x, f_mass(x), color="red")
        return f_mass

    def get_turnoff_mass(self, log_age):
        """
        Determine the turnoff mass for a given age

        Args:
            log_age: float
        Returns: 
            Turnoff mass
        """
        # Restrict the models to the given age
        isoc_age = self.select_age(log_age)
        # Get the turnoff mass
        return np.max(isoc_age['Mass'][ isoc_age["label"]==1 ])
    
    def get_max_mass(self, log_age):
        """
        Determine the maximum mass for a given age

        Args:
            log_age: float
        Returns: 
            Maximum mass
        """
        # Restrict the models to the given age
        isoc_age = self.select_age(log_age)
        # Get the maximum mass
        return np.max(isoc_age['Mass'])

    def get_turnoff_G(self, log_age):
        """
        Determine the turnoff mass for a given age

        Args:
            log_age: float
        Returns: 
            Turnoff mass
        """
        # Restrict the models to the given age
        isoc_age = self.select_age(log_age)
        # Get the turnoff mass
        return np.min(isoc_age['G_fSBmag'][ isoc_age["label"]==1 ])
