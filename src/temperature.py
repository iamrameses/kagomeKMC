# src/temperature.py

import cupy as cp
from .cubicspline import CpCubicSpline


def temperature_function(duration, temp_initial, temp_final=8.5, nsamples=1e7,  method='linear', cooling_constant=None):
    """Compute cooling curve from an initial and final temperature over some given duration.

    Parameters
    ----------
    duration : float
        Total duration (in seconds) over which to calculate temperatures.
    temp_initial : float
        Initial temperature of the simulation in Kelvin.
    temp_final : float
        Ambient temperature (temperature that simulations cools to) in Kelvin. 
        Default is 9.0 K
    nsamples : int
        Number of samples to use for calculating temperatures. 
        Default is 1e7 samples.
    method : str
        Method to use for calculating cooling constant. 
        Default is 'linear'. Options are 'linear', 'exponential', 'inv_exponential.
    cooling_constant : float
        Cooling constant (in Kelvin/sec) to use for calculating temperatures. 
        If None, the cooling constant will be calculated based on the method.
        Default is None. 

    Returns
    -------
    cstemps : CpCubicSpline
        Cubic spline interpolation of temperatures over time.
    temp_initial : float
        Initial temperature of the simulation in Kelvin.
    temp_final : float
        Ambient temperature (temperature that simulations cools to) in Kelvin.
    """
    # Calculate times array
    times = cp.linspace(0, duration+(0.5*duration), int(nsamples))
    if method == 'linear':
        if cooling_constant is None:
            # Calculate cooling constant
            cooling_constant = (temp_initial - temp_final) / duration
        # Calculate temperatures using a linear function
        temps = temp_initial - cooling_constant * times
    elif method == 'exponential':
        if cooling_constant is None:
            cooling_constant = cp.log(temp_initial - temp_final) / duration
        # Calculate temperatures using Newton's Law of Cooling
        temps = temp_final + (temp_initial - temp_final) * cp.exp(-cooling_constant * times)
    elif method == 'inv_exponential':
        if cooling_constant is None:
            cooling_constant = cp.log((temp_initial - temp_final)*(2*duration)) / duration
        # Calculate temperatures using inverse Newton's Law of Cooling
        temps = temp_initial - (1/(2*duration)) * cp.exp(cooling_constant * times)
    # Return cubic spline interpolation of temperatures
    tempfunction = CpCubicSpline(times, temps)
    return tempfunction, temp_initial, temp_final