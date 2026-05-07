# Numerical Calculation of Mass and Radius for an EOS of Neutron Stars
The library includes a 4th order Runga Kutta
method to numerically solve the Relativistic Hydrodynamical 
Equations given an Equation of State. 

Following Oppenheimer & Volkoff (1939) and Chandrasekher (1957)
the equations are re-written in terms of dimensionless quantities.

The Equations of States (EOS) used here were derived by Wiringa et al (1988) and the fit
functions were given by Kutschera & Kotlorz (1993).

New EOS can be replaced with the ones presented here and the numerical 
calculating functions should be self consistent. It is critical, however, that the EOS are 
a function of the baryon density and the initial conditions such as initial radius, mass, and
baryon density are specified. Tabulated EOS can be added through interpolator.py and adding a new index. 

EOSNeutronStars.py is the primary module which includes the various required functions to perform the numerical calculations.

Make sure to install the following packages:

  - numpy

  - matplotlib

  - math

Feel free to contact me regarding any questions at vsudhakar7@gatech.edu
