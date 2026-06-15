import logging
logging.basicConfig(level='INFO')
import numpy as np

from astropy import units as u
from astropy import constants as c
from astropy.stats import signal_to_noise_oir_ccd
from astroquery.vizier import Vizier
from astroquery.xmatch import XMatch

from scipy.interpolate import PchipInterpolator

from synphot import (Empirical1D,
                     Observation,
                     SourceSpectrum,
                     SpectralElement,
                     units) 


class InterpolatableTable(object):
    '''
    Provide smoothed lookup tables for user-supplied input data.
    '''
    def __init__(self, x, y, clamp_edges=False):
        '''
        Parameters
        ----------
        x : np.ndarray
            Array of abscissa
        y : np.ndarray
            Array of ordinate
        clamp_edges : bool (optional)
            If True, add data points to the beginning and end of x for which
            y = 0.
        '''
        assert x.shape == y.shape
        self.x = x
        self.y = y

        if clamp_edges:
            # Useful if input filter data might cause interpolation into negative transmissions, for example.
            # Clamp unknown values to zero.
            dx = np.diff(x).mean()
            self.x = np.concatenate([[x[0] - (2 * dx), x[0] - dx], x, [x[-1] + dx, x[-1] + (2 * dx)]])
            self.y = np.zeros(4 + len(y))
            self.y[2:-2] = y

    def interp(self, x_new):
        '''
        Return the interpolated values at the supplied x-coordinates

        Parameters
        ----------
        x_new : float, np.ndarray
            Query value(s) for interpolation

        Returns
        -------
        Interpolated value(s)
        '''
        pchip_interp = PchipInterpolator(self.x, self.y)
        return pchip_interp(x_new)


class Sensor(object):
    '''
    Container object for properties related to CCD/CMOS imagers
    '''
    def __init__(self, shape, px_size, dark_current, read_noise, full_well, bits, gain_adu_per_e=1./u.electron, qe_table=None, name=None, cost=0):
        '''
        Parameters
        ----------
        shape : tuple
            2-element tuple describing the number of pixels along each sensor
            axis
        px_size : astropy.Quantity, length
            Assuming square pixels, the side length of one pixel
        dark_current : astropy.Quantity, e-/s
            Sensor dark current, in e-/s
        read_noise : astropy.Quantity
            Sensor read noise per exposure, in e-
        full_well : astropy.Quantity
            Sensor full well capacity in e-/pix (base gain)
        gain_adu_per_e : astropy.Quantity (optional)
            Conversion factor between electrons and ADC data numbers, units
            1/e-. If not supplied, default is 1:1.
        qe_table : InterpolatableTable (optional)
            Lookup table for sensor quantum efficiency. All sensors assumed
            mono. If not supplied, default is 1.0 from 400-1100 nm.
        name : str (optional)
            Name for printing. If not supplied, default is blank str.
        cost : float (optional)
            Cost for summing into system cost. If not supplied, default is 0.
        '''
        self.shape = shape * u.pix
        assert len(self.shape) == 2
        self.px_size = px_size.to(u.micron)
        self.dark_current = dark_current.to(u.electron / u.s)
        self.read_noise = read_noise.to(u.electron)
        self.full_well = full_well.to(u.electron / u.pix)
        self.gain_adu_per_e = gain_adu_per_e.to(1. / u.electron)

        self.megapixels = self.shape[0] * self.shape[1] / 1e6 / u.pix**2
        self.sensor_size = (self.shape * self.px_size / u.pix).to(u.mm)
        self.bits = bits
        # Assume no fancy bit-packing to store data on disk:
        # store the integer number of bytes required to represent the ADC data.
        num_bytes_disk = np.ceil(bits / 8)
        self.frame_storage_size = num_bytes_disk * 8 * self.megapixels * 1e6

        if qe_table is None:
            logging.warning('No quantum efficiency curve supplied. Assuming unity QE 400-1100 nm.')
            x = np.arange(400, 1100, 100) * u.nm
            y = np.ones_like(x.value)
            self.qe_table = InterpolatableTable(x, y)
        else:
            self.qe_table = qe_table

        if name is None:
            self.name = ''
        else:
            self.name = name

        self.cost = cost

    def __repr__(self):
        return self.name

    def qe(self, lambd):
        '''
        Return the interpolated quantum efficiency values at the supplied
        wavelength coordinates

        Parameters
        ----------
        lambd : astropy.Quantity, must match units QE was initialized with
            Wavelength coordinates for lookup

        Returns
        -------
        Interpolated value(s)
        '''
        return self.qe_table.interp(lambd)


class Lens(object):
    '''
    Container object for properties related to focusing optics.
    '''
    def __init__(self, focal_length, aperture_diam, tau_table=None, aberration_multiplier=10, name=None, cost=0):
        '''
        focal_length : astropy.Quantity, length
            Optic focal length
        aperture_diam : astropy.Quantity, length
            Optic aperture
        tau_table : InterpolatableTable (optional)
            Transmission vs. wavelength table. For instance, this would include
            glass absorption and coating transmission, if available. If not
            supplied, default is 1.0 from 400-1100 nm.
        aberration_multiplier : float (optional)
            Multiply the diffraction-limited PSF by a factor to account for
            imperfect optics. Refractors (like commercial lenses) are not
            diffraction-limited. In TIM ground-based testing, we achieved typical
            PSFs ~13x the diffraction limit, in average seeing. In TIMcam at
            float, we achieved typical PSFs ~7-13x the diffraction limit,
            focus-dependent.
        name : str (optional)
            Name for printing. If not supplied, default is blank str.
        cost : float (optional)
            Cost for summing into system cost. If not supplied, default is 0.
        '''
        self.f = focal_length.to(u.mm)
        self.d = aperture_diam.to(u.mm)
        self.aberration_multiplier = aberration_multiplier

        self.area = np.pi * (self.d / 2.) ** 2

        if tau_table is None:
            logging.warning('No optical transmission curve supplied. Assuming unity tau 400-1100 nm.')
            x = np.arange(400, 1100, 100) * u.nm
            y = np.ones_like(x.value)
            self.tau_table = InterpolatableTable(x, y)
        else:
            self.tau_table = tau_table

        if name is None:
            self.name = ''
        else:
            self.name = name

        self.cost = cost

    def __repr__(self):
        return self.name

    def fwhm(self, lambd):
        '''
        Calculate the full width at half maximum of the point spread function,
        as a function of wavelength.

        Parameters
        ----------
        lambd : astropy.Quantity (length)
            Operating wavelength(s)

        Returns
        -------
            1.029 * aberration_multiplier * \lambda / D, converted to arcsec
        '''
        return (1.029 * u.rad * lambd.to(u.m) / self.d.to(u.m)).to(u.arcsec) * self.aberration_multiplier

    def airy_diam(self, lambd):
        '''
        Calculate the diameter of the first airy null as a function of
        wavelength.

        Parameters
        ----------
        lambd : astropy.Quantity (length)
            Operating wavelength(s)

        Returns
        -------
            2 * 1.22 * aberration_multiplier * \lambda / D, converted to arcsec
        '''
        return (2 * 1.22 * u.rad * lambd.to(u.m) / self.d.to(u.m)).to(u.arcsec) * self.aberration_multiplier

    def spot_size(self, lambd):
        '''
        Calculate the spot size based on the numerical aperture.

        Parameters
        ----------
        lambd : astropy.Quantity (length)
            Operating wavelength(s)

        Returns
        -------
           Spot size in microns 
        '''
        NA = 1 / (2 * self.f / self.d)
        return (1.22 * lambd / NA).to(u.um) * self.aberration_multiplier

    def tau(self, lambd):
        '''
        Return the interpolated transmission values at the supplied wavelength
        coordinates

        Parameters
        ----------
        lambd : astropy.Quantity, must match units tau_table was initialized with
            Wavelength coordinates for lookup

        Returns
        -------
        Interpolated value(s)
        '''
        return self.tau_table.interp(lambd)


class Filter(object):
    '''
    Container for quantities related to band-defining filters.
    '''
    def __init__(self, zero_point_flux, tau_table=None, name=None, cost=0):
        '''
        zero_point_flux : astropy.Quantity
            Flux of a 0th-magnitude star in the given filter, in W cm^-2 um^-1.
            This should be the average flux from a transmission-weighted
            integral over the reference star spectrum. For help finding or
            calculating zero-point fluxes for a given filter, see the Spanish
            Virtual Observatory Filter Profile Service:
            https://svo2.cab.inta-csic.es/theory/fps/
        tau_table : InterpolatableTable (optional)
            Wavelength-dependent filter transmission curve. If not supplied,
            default is 1.0 from 400-1100 nm.
        name : str (optional)
            Name for printing. If not supplied, default is blank str.
        cost : float (optional)
            Cost for summing into system cost. If not supplied, default is 0.
        '''
        # the flux in W cm^-2 um^-1 of a magnitude 0 star
        self.zero_point_flux = zero_point_flux
        if tau_table is None:
            x = np.arange(400, 1100, 100) * u.nm
            y = np.ones_like(x.value)
            self.tau_table = InterpolatableTable(x, y)
        else:
            self.tau_table = tau_table

        self.cost = cost

        if name is None:
            self.name = ''
        else:
            self.name = name

    def __repr__(self):
        return self.name

    def tau(self, lambd):
        '''
        Return the interpolated transmission values at the supplied wavelength
        coordinates

        Parameters
        ----------
        lambd : astropy.Quantity, must match units tau_table was initialized with
            Wavelength coordinates for lookup

        Returns
        -------
        Interpolated value(s)
        '''
        return self.tau_table.interp(lambd)


class StarCamera(object):
    '''
    Provide quantities that are derived from component parts of the star camera
    system.
    '''
    def __init__(self, sensor: Sensor, lens: Lens, filter: Filter, name=None):
        '''
        sensor : Sensor
        lens : Lens
        filter : filter
        name : str (optional)
            Name for printing. If not supplied, default is blank str.
        '''
        self.sensor = sensor
        self.lens = lens
        self.filter = filter
        
        if name is None:
            self.name = ''
        else:
            self.name = name

        self.cost = sensor.cost + lens.cost + filter.cost

        self.fov = 2 * np.arctan(sensor.sensor_size / 2 / lens.f).to(u.deg)
        self.plate_scale = (self.fov / sensor.shape).to(u.arcsec / u.pix)

    def __repr__(self):
        return self.name

    def get_mean_lambd(self, lambd):
        '''
        Filter-weighted mean wavelength

        Parameters
        ----------
        lambd : astropy.Quantity, must match units filter was initialized with
            Wavelength coordinates for lookup
        
        Returns
        -------
        Filter-weighted mean wavelength
        '''
        return np.trapezoid(lambd * self.filter.tau(lambd), lambd) / np.trapezoid(self.filter.tau(lambd), lambd)

    def get_response(self, lambd):
        '''
        System response curve, the product of quantum efficiency, optics
        transmission, and filter bandpass.

        Parameters
        ----------
        lambd : astropy.Quantity, must match units filter, QE, and lens
            transmission were initialized with
            Wavelength coordinates for lookup

        Returns
        -------
        Total response value(s), evaluated at lambd
        '''
        return self.sensor.qe(lambd) * self.lens.tau(lambd) * self.filter.tau(lambd)


# helpers
def get_optics_transmission(lambd):
    '''
    Generate an array of 0.9 with shape matching lambd.

    Parameters
    ----------
    lambd : astropy.Quantity
    '''
    return np.ones_like(lambd.value) * 0.9


def get_filter_transmission(lambd, center=650*u.nm, center2=None, width=7*u.nm, max_transmission=0.9):
    '''
    Generate a scaled and shifted sigmoid function representing a long-pass or
    bandpass filter profile. Short-pass or bandstop filters could be generated
    with 1-this profile.

    Parameters
    ----------
    lambd : astropy.Quantity, length
        Wavelength coordinates for lookup
    center : astropy.Quantity, length (optional)
        Filter cut-on wavelength, specified as 50% transmission. If not
        supplied, default is 650 nm.
    center2 : astropy.Quantity, length (optional)
        Filter cut-off wavelength, specified as 50% transmission. If not
        supplied, default is None, for a long-pass filter. This function has not
        been tested with center2 < center.
    width : astropy.Quantity, length, (optional)
        Filter cut on/off transition width. Larger values give a smoother cut
        on/off. If not supplied, default is 7 nm.
    max_transmission : float
        Value that all returned values are multiplied by, representing the
        maximum value of the transmission curve. If not supplied, default is 0.9

    Returns
    -------
    lambd-shaped array of transmission values
    '''
    scale = 1. / width
    if center2 is None:
        return max_transmission * (1. / (1. + np.exp(-scale * (lambd - center))))
    else: # Hack to test narrowband filters
        cut0 = (1. / (1. + np.exp(-scale * (lambd - center))))
        cut1 = 1 - (1. / (1. + np.exp(-scale * (lambd - center2))))
        return max_transmission * cut0 * cut1


def get_sky_brightness(lambd, scale_factor=0.351, altitude=35*u.km):
    '''
    This is a coarse, smoothed model of the atmosphere spectrum based on Fig. 4
    of Alexander+ 1999's MODTRAN spectrum, suitable for rough estimates of sky
    brightness. It does not include any absorption or emission features. It is
    defined from 400-1200 nm. A scale_factor of 1 and altitude of 35 km 
    reproduces Fig. 4. Alternative altitudes scale the spectrum up or down
    according to Alexander+ 1999's equation.

    Parameters
    ----------
    lambd : astropy.Quantity, length
        Wavelength coordinate(s) for lookup
    scale_factor : float (optional)
        Multiply the spectrum by this value. This is an easy way to simulate
        brighter or dimmer sky conditions. If no value is supplied, the
        normalization will reproduce the counts seen by TIMCam in 2025 at
        37.8 km.
    altitude : astropy.Quantity, length (optional)
        Scales the spectrum up or down for lower or higher altitudes. If no
        value is supplied, the default is 35 km.

    Returns
    -------
    The sky specific intensity values, in W cm^-2 um^-1 sr^-1 for each lambd
    value
    '''
    # A scale factor of 0.351 is required to match the photon counts of the
    # TIMCam 2025 piggyback flight with the predictions of the TIMCam hardware
    # model.
    # Alexander 1999, Fig. 4
    # MODTRAN, 35km, 30deg SZA
    # note: sky brightness may actually go UP past 1200 nm! water, etc.
    plot_lambd = np.array([400, 500, 600, 700, 800, 900, 1000, 1100, 1200]) * u.nm
    plot_val = 1e-5 * np.array([10., 5., 2.2, 1., 0.5, 0.2, 0.1, 0.0, 0.0]) * u.W / u.cm**2 / u.sr / u.um
    interp = PchipInterpolator(plot_lambd, plot_val)
    # Alexander+ 1999, Fig. 4
    scale_height = 7.348 * u.km
    literature_alt = 35 * u.km
    altitude_scale_factor = np.exp(-(altitude - literature_alt) / scale_height)
    return scale_factor * altitude_scale_factor * interp(lambd) * plot_val.unit 


def calc_limiting_mag(snr_limit, mags, snrs):
    '''
    Use an InterpolatableTable to back-interpolate a limiting magnitude from
    an SNR-magnitude relation.
    '''
    snr_table = InterpolatableTable(snrs, mags)
    return snr_table.interp(snr_limit)


def get_tycho_stars(coord, width, height, mag_limit=11, nlimit=1000):
    '''
    Query the online Vizier Tycho2 database for a region of interest. Sorted by
    VTmag.

    Parameters
    ----------
    coord : astropy.coordinate.SkyCoord
    width : astropy.Quantity, angle
    height : astropy.Quantity, angle
    mag_limit : float
        Limiting Tycho2 magnitude. Default is 11, which is similar to the limit
        for the constituents of astrometry.net 4100-series files.
    nlimit : int
        Limit the length of returned values

    Returns
    -------
    astropy.Table instance with query results
    '''
    query = Vizier(
        columns=['RAmdeg', 'DEmdeg', 'BTmag', 'VTmag', '_RAJ2000', '_DEJ2000'],#, 'RA(ICRS)', 'DE(ICRS)'],
        column_filters={'VTmag' : f'<{mag_limit}'},
        row_limit=1000,
        timeout=240,
    )
    logging.debug(f'Querying Vizier service... {coord}, {width}x{height}')
    table = query.query_region(coord, width=width, height=height, catalog=['I/259/tyc2'])
    t = table[table.keys()[0]]
    t.sort('VTmag')
    return t[:nlimit]


def get_median_Teff(coord, width, height, match_radius=1*u.arcsec):
    '''
    Query the Teff info associated with a region of sky. First, the Tycho
    catalog is queried to get a list of candidate stars that would appear in an
    astrometry.net database. Then, these stars are cross-matched with the Gaia
    DR3 database to retrieve their effective tmperatures.

    Parameters
    ----------
    coord : astropy.coordinate.SkyCoord
    width : astropy.Quantity, angle
    height : astropy.Quantity, angle
    match_radius : astropy.Quantity, angle
        Thershold for considering coordinates of objects in each database to be
        a match.
    nlimit : int
        Limit the length of returned values

    Returns
    -------
    median effective temperature: astropy.Quantity, K
    astropy.Table entry with all retrieved Teffs in the FoV belonging to Tycho2
        stars
    astropy.Table instance with query results, including the full Gaia DR3
        xmatch results, which we use later to get the RPmag values.
    '''
    tycho_stars = get_tycho_stars(coord, width, height, mag_limit=11, nlimit=1000)
    # Get a list of Gaia Teffs by cross-referencing the given ra/dec
    gaia_dr3 = 'vizier:I/355/gaiadr3'
    logging.debug(f'Querying xMatch service... {coord}, {width}x{height}, r={match_radius}, {gaia_dr3}')
    table = XMatch.query(
        cat1=tycho_stars,
        colRA1='_RAJ2000',
        colDec1='DEmdeg',
        cat2=gaia_dr3,
        max_distance=match_radius
    )
    nonzero = table['Teff'] > 0
    Teffs = table['Teff'][nonzero]
    try:
        Teffs = Teffs.to(u.K)
    except AttributeError as e:
        Teffs *= u.K
    except Exception as e:
        raise e
    return (np.median(Teffs), Teffs, table)


def get_model_flux_density(Teff):
    '''
    Returns a blackbody flux density curve, evaluatable over wavelength.
    This curve is normalized to a solid angle of 1 sr, so it must be
    renormalized according to the magnitude of the star providing the Teff.

    Parameters
    ----------
    Teff : astropy.Quantity, K
        Stellar effective temperature

    Returns
    -------
    model_curve : lambda function
        Call with a wavelength or array of wavelengths to evaluate the 
        normalized blackbody curve.
    '''

    def b(lambd, T):
        # wien
        num = 2. * c.h * c.c ** 2.
        expn = c.h * c.c / (lambd * c.k_B * T)
        return num / ((lambd ** 5.) * (np.exp(expn) - 1.)) / u.sr
    # Because we only care about the shape of the curve, we can arbitrarily
    # multiply by any angular size we want - we don't know how far away a
    # given star is, and we don't care because this curve will rescaled later
    # anyway.
    res = lambda lambd: b(lambd, Teff).to(u.W / u.cm**2 / u.um / u.sr)
    model_curve = lambda lambd: (res(lambd) / np.nanmax(res(lambd)).value) * u.sr
    # Now do you see why I call all of my wavelength arrays lambd?
    return model_curve


def get_zero_point_flux(lambd, filter: Filter):
    '''
    To generate a zero-point flux, observe a 0mag star (Vega) in the given
    filter.
    Following the SVO routine,
    F0 = \int(T(\lambda) Vega(\lambda) d\lambda) / \int(T(\lambda) d\lambda)

    Parameters
    ----------
    lambd : astropy.Quantity, length
        Wavelength coordinate(s) for lookup
    filter: Filter

    Returns
    -------
    Filter zero-point flux, in W cm^-2 um^-1
    '''
    vega = SourceSpectrum.from_vega()
    spec_vega = vega(lambd.to(u.AA), flux_unit=units.FLAM)
    numer = np.trapezoid(spec_vega * filter.tau(lambd), lambd)
    denom = np.trapezoid(filter.tau(lambd), lambd)
    F0 = (numer/denom).to(u.W / u.cm**2 / u.um)
    return F0


def get_equivalent_mag(lambd, ref_mag, ref_filt, new_filt, model_flux_density):
    '''
    Use synthetic photometry to calculate the magnitude of a star in one filter
    system, given the magnitude and filter bandpass in another system.

    Parameters
    ----------
    lambd : astropy.Quantity
        Increasing array of wavelengths to evaluate curves at
    ref_mag : float or np.ndarray
        magnitude in reference filter system
    ref_filt : function
        Evaluatable over lambd, the reference filter system bandpass
    new_filt : function
        Evaluatable over lambd, the new filter system bandpass
    model_flux_density : function
        Evaluatable over lambd, the flux density model for the star in question,
        in or convertible to W/cm^2/um.

    Returns
    -------
    magnitude in new filter system (VEGA magnitudes)
    '''
    def core(lambd, ref_mag, ref_filt, new_filt, model_flux_density):
        '''
        Helper to encapsulate synphot stuff, to allow broadcasting one level
        above.
        '''
        # Load in the model spectrum from Teff and Planck law
        src_norm = SourceSpectrum(
            Empirical1D,
            points=lambd.to(u.AA),
            lookup_table=model_flux_density(lambd).to(u.W / u.cm**2 / u.AA)
        )
        
        # Renormalize the curve, given the reference magnitude, Vega spectrum, and
        # reference system filter bandpass.
        # This gives the star the correct spectral content and amplitude to
        # reproduce the given magnitude in the given system.
        bandpass_ref = SpectralElement(
            Empirical1D,
            points=lambd.to(u.AA),
            lookup_table=ref_filt(lambd)
        )
        src = src_norm.normalize(ref_mag * units.VEGAMAG, bandpass_ref, vegaspec=SourceSpectrum.from_vega())

        # Observe the reference spectrum and scaled spectrum in the new filter
        # system
        bandpass_new = SpectralElement(
            Empirical1D,
            points=lambd.to(u.AA),
            lookup_table=new_filt(lambd)
        )
        obs_vega_new = Observation(SourceSpectrum.from_vega(), bandpass_new)
        obs_src_new = Observation(src, bandpass_new)
        # the new magnitude is the ratio of count rates to Vega in the new filter
        # system
        A = 1 * u.cm**2
        counts_vega_new = obs_vega_new.countrate(A)
        counts_src_new = obs_src_new.countrate(A)
        return -2.5 * np.log10(counts_src_new / counts_vega_new)

    # Handle user passing an array of magnitudes
    if np.isscalar(ref_mag):
        return core(lambd, ref_mag, ref_filt, new_filt, model_flux_density)
    else:
        args = [
            [lambd] * len(ref_mag),
            ref_mag,
            [ref_filt] * len(ref_mag),
            [new_filt] * len(ref_mag),
            [model_flux_density] * len(ref_mag),
        ]
        return np.array(list(map(core, *args)))


def electrons_per_sec_spectral(tau, eta, A_tel, lambd, flux):
    '''Calculate the signal level per pixel, in electrons/s, given the input 
    telescope and target parameters.
    
    Following McLean 2008, "Electronic Imaging in Astronomy," Ch. 9.

    Parameters
    ----------
    tau : float
        The total transmittance due to all lossy optics in the path, e.g. 
        mirror reflectivity, coatings, filters.
    eta : float
        The quantum efficiency of the detector.
    A_tel : astropy.Quantity, area
        The unobscured area of the telescope, in meters squared.
    lambd : astropy.Quantity, length
        The wavelength array being integrated over, in meters
    flux : astropy.Quantity
        The absolute flux in W cm^-2 um^-1

    Returns
    -------
    float
        Amount of signal in each pixel, in electrons/s
    '''
    return u.electron * (1. / (c.h * c.c)) * A_tel * np.trapezoid(lambd * eta * tau * flux, lambd)


def simple_snr_spectral(
    t,
    lambd,
    target_mag,
    starcam: StarCamera,
    min_aperture_area=4,
    sky_brightness_factor=.263,
    altitude=35*u.km,
    return_components=False
):
    '''
    Parameters
    ----------
    t : astropy.Quantity, time
        Exposure time
    lambd : astropy.Quantity, length
        Wavelength(s) to evaluate wavelength-dependent quantities at
    target_mag : float
        magnitude of target in filter for which the zero point flux is given
    starcam: StarCamera
    min_aperture_area (optional): int
        For optics with diffraction-limited resolution finer than the plate
        scale (undersampled), use this number of pixels as the minimum number
        of pixels for calculating the sky noise contribution. 4 is the default
        because a star that falls at the corner of a pixel will have its light
        spread evenly across a 4 pixel area. This also helps account for
        possible scattering, motion blur, and defocus.
    sky_brightness_factor : float (optional)
        Multiply the spectrum of the sky by this factor to explore different sun-relative pointings/altitudes.
        A factor of 0.263 and altitude 37 km reproduces the background counts measured by the TIM SC2 during Fort Sumner 2024.
        A factor of 0.351 and altitude 37.8 km reproduces the background counts measured by the TIMcam piggyback during Fort Sumner 2025.
        A factor of 1 and altitude 35 km reproduces the spectrum from Alexander+1999 Fig.4 MODTRAN.
    altitude : astropy.Quantity, length (optional)
        Scale the sky irradiance spectrum according to Alexander+1999
    return_components : bool (optional)
        If supplied, return components of the SNR calculation and flags for
        saturation due to star, sky, or sky + star.

    Returns
    -------
    float
        Signal-to-noise ratio in a single exposure
    source_electrons_per_sec : astropy.Quantity, e-/s (optional)
        Total e-/s in aperture due to source
    sky_electrons_per_sec : astropy.Quantity, e-/s (optional)
        Total e-/s in aperture due to sky
    aperture_area_px : astropy.Quantity, pix^2 (optional)
        Number of pixels contributing to source and sky counts
    warned_source_sat : bool (optional)
        Sensor saturation is expected due to star counts alone
    warned_sky_sat : bool (optional)
        Sensor saturation is expected due to star counts alone
    warned_sum_sat : bool (optional)
        Sensor saturation is expected due to sum of star and sky counts
    '''
    lambd = lambd.to(u.nm)
    D = starcam.sensor.dark_current
    R = starcam.sensor.read_noise
    plate_scale_arcsec_per_px = starcam.plate_scale.mean().to(u.arcsec / u.pix)
    psf_diam = starcam.lens.fwhm(starcam.get_mean_lambd(lambd))

    # Source signal
    flux = starcam.filter.zero_point_flux.to(u.W / u.cm**2 / u.nm) * (10. ** (-0.4 * target_mag))
    source_electrons_per_sec = electrons_per_sec_spectral(
        starcam.lens.tau(lambd) * starcam.filter.tau(lambd), # total optical transmission
        starcam.sensor.qe(lambd),
        starcam.lens.area,
        lambd,
        flux
    ).decompose()

    # Model gives W/cm^2/sr/um, want sky signal in 1 arcsec^2:
    sky_flux = get_sky_brightness(lambd, scale_factor=sky_brightness_factor, altitude=altitude).to(u.W / u.cm**2 / u.arcsec**2 / u.nm ) * u.arcsec**2
    sky_electrons_per_sec_per_sq_arcsec = electrons_per_sec_spectral(
        starcam.lens.tau(lambd) * starcam.filter.tau(lambd), # total optical transmission
        starcam.sensor.qe(lambd),
        starcam.lens.area,
        lambd,
        sky_flux
    ).decompose() / u.arcsec**2

    # fake aperture photometry: all source photons end up inside the aperture

    aperture_area_px = np.pi * (psf_diam.to(u.arcsec) / plate_scale_arcsec_per_px.to(u.arcsec / u.pix) / 2.)**2
    logging.debug(f'{starcam.name} aperture area px: {aperture_area_px}')
    if min_aperture_area > 0:
        # the aperture is matched to the greater of the two: PSF or the minimum.
        # if seeing is to be considered, it shall be considered in the psf diameter.
        min_area_px = min_aperture_area * u.pix**2
        aperture_area_px = np.clip(aperture_area_px, min_area_px, None)
        logging.debug(f'aperture area clipped to {aperture_area_px}')

    sky_electrons_per_sec_per_px = sky_electrons_per_sec_per_sq_arcsec * (plate_scale_arcsec_per_px ** 2)

    warned_source_sat = False
    warned_sky_sat = False
    warned_sum_sat = False

    source_electrons_per_px_per_exposure = source_electrons_per_sec * t / aperture_area_px * u.pix
    if source_electrons_per_px_per_exposure > starcam.sensor.full_well:
        logging.warning(
            f'SOURCE SATURATION: electron count from source mag {target_mag} exceeds sensor full well current, saturated!\n' +
            f'{source_electrons_per_px_per_exposure:.0f} > {starcam.sensor.full_well:.0f}\n' +
            f'sensor: {starcam.sensor.name}, lens: {starcam.lens.name}, filter: {starcam.filter.name}'
        )
        warned_source_sat = True

    sky_electrons_per_px_per_exposure = sky_electrons_per_sec_per_px * t * u.pix
    if sky_electrons_per_px_per_exposure > starcam.sensor.full_well:
        logging.warning(
            f'SKY SATURATION: electron count from sky exceeds sensor full well current, saturated!\n' +
            f'{sky_electrons_per_px_per_exposure:.0f} > {starcam.sensor.full_well:.0f}\n' + 
            f'sensor: {starcam.sensor.name}, lens: {starcam.lens.name}, filter: {starcam.filter.name}'
        )
        warned_sky_sat = True

    if source_electrons_per_px_per_exposure + sky_electrons_per_px_per_exposure > starcam.sensor.full_well:
        logging.warning(
            f'SUM SATURATION: sum of sky and source counts exceeds sensor full well current, saturated!\n' +
            f'{source_electrons_per_px_per_exposure + sky_electrons_per_px_per_exposure:.0f} > {starcam.sensor.full_well:.0f}\n' + 
            f'sensor: {starcam.sensor.name}, lens: {starcam.lens.name}, filter: {starcam.filter.name}'
        )
        warned_sum_sat = True

    sky_electrons_per_sec = sky_electrons_per_sec_per_px * aperture_area_px

    logging.debug(f'plate scale: {starcam.plate_scale}')
    logging.debug(f'psf diameter: {psf_diam.to(u.arcsec)}')
    logging.debug(f'source e-/s: {source_electrons_per_sec}')
    logging.debug(f'sky e-/s/px: {sky_electrons_per_sec_per_px}')
    logging.debug(f'sky e-/s: {sky_electrons_per_sec}')

    snr = signal_to_noise_oir_ccd(
        t,
        source_electrons_per_sec / u.electron, # really wants 1/s
        sky_electrons_per_sec_per_px / u.electron * u.pix**2,
        D / u.electron,
        R / u.electron,
        aperture_area_px.value,
        gain=1 # all values in electrons already
    )
    if return_components:
        return snr, (source_electrons_per_sec, sky_electrons_per_sec, aperture_area_px), (warned_source_sat, warned_sky_sat, warned_sum_sat)
    return snr