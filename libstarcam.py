import logging
logging.basicConfig(level='INFO')
import numpy as np

from astropy import units as u
from astropy import constants as c
import astropy.coordinates as coord
from astropy.stats import signal_to_noise_oir_ccd
from astroquery.vizier import Vizier
from astroquery.xmatch import XMatch

from scipy.interpolate import PchipInterpolator


class InterpolatableTable(object):
    def __init__(self, x, y, clamp_edges=False):
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
        pchip_interp = PchipInterpolator(self.x, self.y)
        return pchip_interp(x_new)


class Sensor(object):
    def __init__(self, shape, px_size, dark_current, read_noise, full_well, bits, gain_adu_per_e=1./u.electron, qe_table=None, name=None, cost=0):
        self.shape = shape * u.pix
        self.px_size = px_size
        self.dark_current = dark_current
        self.read_noise = read_noise
        self.full_well = full_well
        self.gain_adu_per_e = gain_adu_per_e

        self.megapixels = self.shape[0] * self.shape[1] / 1e6 / u.pix**2
        self.sensor_size = (self.shape * self.px_size / u.pix).to(u.mm)
        self.bits = bits
        # Assume no fancy bit-packing to store data on disk:
        # store the integer number of bytes required to represent the ADC data.
        num_bytes_disk = np.ceil(bits / 8)
        self.frame_storage_size = num_bytes_disk * 8 * self.megapixels * 1e6

        if qe_table is None:
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

    def qe(self, lambd):
        return self.qe_table.interp(lambd)


class Lens(object):
    def __init__(self, focal_length, aperture_diam, tau_table=None, name=None, cost=0):
        self.f = focal_length
        self.d = aperture_diam

        self.area = np.pi * (self.d / 2.) ** 2

        if tau_table is None:
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

    def fwhm(self, lambd):
         return (1.029 * u.rad * lambd.to(u.m) / self.d.to(u.m)).to(u.arcsec)

    def airy_diam(self, lambd):
        return (2 * 1.22 * u.rad * lambd.to(u.m) / self.d.to(u.m)).to(u.arcsec)

    def spot_size(self, lambd):
        NA = 1 / (2 * self.f / self.d)
        return (1.22 * lambd / NA).to(u.um)

    def tau(self, lambd):
        return self.tau_table.interp(lambd)


class Filter(object):
    def __init__(self, zero_point_flux, tau_table=None, cost=0, name=None):
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

    def tau(self, lambd):
        return self.tau_table.interp(lambd)


class StarCamera(object):
    def __init__(self, sensor: Sensor, lens: Lens, filter: Filter, name=None):
        self.sensor = sensor
        self.lens = lens
        self.filter = filter
        
        if name is None:
            self.name = 'SC'
        else:
            self.name = name

        self.cost = sensor.cost + lens.cost + filter.cost

        self.fov = 2 * np.arctan(sensor.sensor_size / 2 / lens.f).to(u.deg)
        self.plate_scale = (self.fov / sensor.shape).to(u.arcsec / u.pix)

    def get_mean_lambd(self, lambd):
        # filter-weighted mean wavelength
        return np.trapz(lambd * self.filter.tau(lambd), dx=np.diff(lambd).mean()) / np.trapz(self.filter.tau(lambd), dx=np.diff(lambd).mean())
    
    def get_response(self, lambd):
        return self.sensor.qe(lambd) * self.lens.tau(lambd) * self.filter.tau(lambd)


# helpers
def get_optics_transmission(lambd):
    # optical system total transmission, sans filters
    return np.ones_like(lambd.value) * 0.9


def get_filter_transmission(lambd, center=650*u.nm, center2=None, width=7*u.nm, max_transmission=0.9):
    # scaled and shifted sigmoid
    scale = 1. / width
    if center2 is None:
        return max_transmission * (1. / (1. + np.exp(-scale * (lambd - center))))
    else: # Hack to test narrowband filters
        cut0 = max_transmission * (1. / (1. + np.exp(-scale * (lambd - center))))
        cut1 = max_transmission - max_transmission * (1. / (1. + np.exp(-scale * (lambd - center2))))
        return cut0 * cut1 / (np.max(cut0*cut1))


def get_sky_brightness(lambd, scale_factor=0.263, altitude=35*u.km):
    # A scale factor of 0.263 is required to match the ADUs measured by SC2 in
    # the Fort Sumner test flight with the predictions of the SC2 hardware
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
    snr_table = InterpolatableTable(snrs, mags)
    return snr_table.interp(snr_limit)


def get_tycho_stars(coord, width, height, mag_limit=9, nlimit=2000):
    query = Vizier(
        columns=['RAmdeg', 'DEmdeg', 'BTmag', 'VTmag', '_RAJ2000', '_DEJ2000'],#, 'RA(ICRS)', 'DE(ICRS)'],
        column_filters={'VTmag' : f'<{mag_limit}'},
        row_limit=10000,
        timeout=120,
    )
    logging.info(f'Querying Vizier service... {coord}, {width}x{height}')
    table = query.query_region(coord, width=width, height=height, catalog=['I/259/tyc2'])[0]
    table.sort('VTmag')
    return table[:nlimit]


def get_median_Teff(coord, width, height, match_radius=1*u.arcsec, return_array=False):
    tycho_stars = get_tycho_stars(coord, width, height, mag_limit=9, nlimit=10000)
    # Get a list of Gaia Teffs by cross-referencing the given ra/dec
    gaia_dr3 = 'vizier:I/355/paramp'
    logging.info(f'Querying xMatch service... {coord}, {width}x{height}, r={match_radius}, {gaia_dr3}')
    table = XMatch.query(
        cat1=tycho_stars,
        colRA1='_RAJ2000',
        colDec1='DEmdeg',
        cat2=gaia_dr3,
        max_distance=match_radius
    )
    nonzero = table['Teff'] > 0
    Teffs = table['Teff'][nonzero]
    if return_array:
        return np.median(Teffs) * u.K, Teffs * u.K
    return np.median(Teffs) * u.K


def get_model_flux_density(Teff, norm=True):
    # returns a blackbody flux density curve, evaluatable over wavelength
    def b(lambd, T):
        # wien
        num = 2. * c.h * c.c ** 2.
        expn = c.h * c.c / (lambd * c.k_B * T)
        return num / ((lambd ** 5.) * (np.exp(expn) - 1.)) / u.sr
    # Because we only care about the shape of the curve, we can arbitrarily
    # multiply by any angular size we want - we don't know how far away a
    # given star is, and we don't care because this curve will be returned as
    # normalized to peak = 1 and later rescaled anyway.
    res = lambda lambd: b(lambd, Teff).to(u.W / u.cm**2 / u.um / u.sr)
    if norm:
        model_curve = lambda lambd: (res(lambd) / np.nanmax(res(lambd)).value) * u.sr
    else:
        model_curve = lambda lambd: (res(lambd)) * u.sr
    # Now do you see why I call all of my wavelength arrays lambd?
    return model_curve


def get_zero_point_flux(lambd, filter: Filter):
    # Bootstrap off of an existing survey zero-point flux:
    # TYCHO2 V: http://svo2.cab.inta-csic.es/svo/theory/fps/index.php?mode=browse&gname=TYCHO&asttype=
    # Vega system, non-Bessel calibration
    # calibration reference: https://ui.adsabs.harvard.edu/abs/1995A&A...304..110G/abstract
    F0 = (3.99504e-9 * u.erg / u.s / u.cm**2 / u.angstrom).to(u.W / u.cm**2 / u.um)
    # F0 actually has units of spectral flux density, so I am assuming F0 is    
    # actually the average spectral flux density, and F0 * \Delta \lambda, the
    # filter equivalent width, gives the zero-point flux.
    lambd_dat, tau = np.genfromtxt('./TYCHO_TYCHO.V.dat', dtype=float, unpack=True)
    lambd_dat = (lambd_dat * u.angstrom).to(u.nm)
    tycho_v = Filter(F0, tau_table=InterpolatableTable(lambd_dat, tau, clamp_edges=True), name='Tycho V')

    # The zero point flux depends on the spectrum of the reference star, Vega, T_eff=9490K
    # avg of polar and equatorial temps, https://en.wikipedia.org/wiki/Vega
    # # Get median Teff in field
    Teff = 9400 * u.K # this is the BB temp needed to get the Tycho V Vega zero-point flux.
    # Get a normalized spectral flux density curve for that Teff
    model_curve = get_model_flux_density(Teff)(lambd)
    # Scale our "SED" to give the same average spectral flux density as Vega over the V filter
    dl = np.diff(lambd).mean()
    avg_I_lambd = np.trapz(model_curve * tycho_v.tau(lambd), dx=dl) / np.trapz(tycho_v.tau(lambd), dx=dl)
    norm = F0 / avg_I_lambd
    rescaled_curve = model_curve * norm
    # The average flux density of our new curve in our arbitrary filter is our zero-point flux density
    F0_new = np.trapz(rescaled_curve * filter.tau(lambd), dx=dl) / np.trapz(filter.tau(lambd), dx=dl)

    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots()
    # ax.plot(lambd_dat, model_curve)
    # ax.plot(lambd_dat, model_curve * tycho_v.tau(lambd_dat))

    # fig, ax = plt.subplots()
    # ax.plot(lambd, rescaled_curve)
    # ax.plot(lambd, rescaled_curve * filter.tau(lambd))

    return F0_new


def get_equivalent_mag(lambd, ref_mag, ref_resp, new_resp, model_flux_density):
    '''
    Parameters
    ----------
    lambd : np.ndarray
        Increasing array of wavelengths to evaluate curves at
    ref_mag : float
        magnitude in reference filter system
    ref_resp : function
        Evaluatable over lambd, the total response function of the optical system with the reference filter installed
    new_resp : function
        Evaluatable over lambd, the total response function of the optical system with the new filter installed
    model_flux_density : function
        Evaluatable over lambd, the flux density model for the star in question, in or convertible to W/cm^2/um
    '''
    # https://adsabs.harvard.edu/full/1996BaltA...5..459S
    # assume suborbital platform: no atmospheric extinction
    # assume LOS out of galactic plane: negligible reddening/dust extinction
    dl = np.diff(lambd).mean()
    numer = np.trapz(model_flux_density(lambd) * new_resp(lambd), dx=dl).to(u.W / u.cm**2)
    denom = np.trapz(model_flux_density(lambd) * ref_resp(lambd), dx=dl).to(u.W / u.cm**2)
    const = 2.5 * np.log(
        np.trapz(new_resp(lambd), dx=dl) /
        np.trapz(ref_resp(lambd), dx=dl)
    )
    new_mag = -2.5 * np.log(numer / denom) + const + ref_mag
    return new_mag


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
    A_tel : float
        The unobscured area of the telescope, in meters squared.
    lambd : float
        The wavelength array being integrated over, in meters
    flux
        The absolute flux in W / (cm^2 um)
    mag_per : float
        The relative flux in magnitudes of the source.

    Returns
    -------
    float
        Amount of signal in each pixel, in electrons/s
    '''
    return u.electron * (1. / (c.h * c.c)) * A_tel * np.trapz(lambd * eta * tau * flux, dx=lambd.diff().mean())


def simple_snr_spectral(
    t,
    lambd,
    target_mag,
    starcam: StarCamera,
    min_aperture_area=4,
    aberration_multiplier=13,
    sky_brightness_factor=.263,
    altitude=35*u.km,
    return_components=False
):
    '''
    Parameters
    ----------
    t : float
        Exposure time, seconds
    lambd : np.ndarray
        Array of wavelengths to evaluate wavelength-dependent quantities at
    target_mag : float
        magnitude of target in filter for which tel_params['zero_point_flux'] is given
    starcam: StarCamera
    min_aperture_area (optional): int
        For optics with diffraction-limited resolution finer than the plate
        scale (undersampled), use this number of pixels as the minimum number
        of pixels for calculating the sky noise contribution. 4 is the default
        because a star that falls at the corner of a pixel will have its light
        spread evenly across a 4 pixel area. This also helps account for
        possible scattering, motion blur, and defocus.
    aberration_multiplier (optional): float
        Multiply the diffraction-limited PSF by a factor to account for
        imperfect optics. Refractors (like commercial lenses) are not
        diffraction-limited. In TIM ground-based testing, we achieved typical
         PSFs ~13x the diffraction limit, in average seeing.
    altitude : astropy.Quantity, length
        Scale the sky irradiance spectrum according to Alexander+1999
    sky_brightness_factor : float
        Multiply the spectrum of the sky by this factor to explore different sun-relative pointings/altitudes.
        A factor of 0.263 reproduces the background counts measured by the TIM SC2 during Fort Sumner 2024.
        A factor of X and altitude 38 km reproduces the background counts measured by the TIMcam piggyback during Fort Sumner 2025.

    Returns
    -------
    float
        Signal-to-noise ratio in a single exposure
    '''
    D = starcam.sensor.dark_current
    R = starcam.sensor.read_noise
    plate_scale_arcsec_per_px = starcam.plate_scale.mean()
    psf_diam = starcam.lens.fwhm(lambd).mean() * aberration_multiplier

    # Source signal
    flux = starcam.filter.zero_point_flux * (10. ** (-0.4 * target_mag))
    source_electrons_per_sec = electrons_per_sec_spectral(
        starcam.lens.tau(lambd) * starcam.filter.tau(lambd), # total optical transmission
        starcam.sensor.qe(lambd),
        starcam.lens.area,
        lambd,
        flux
    ).decompose()

    # Model gives W/cm^2/sr/um, want sky signal in 1 arcsec^2:
    sky_flux = get_sky_brightness(lambd, scale_factor=sky_brightness_factor, altitude=altitude).to(u.W / u.cm**2 / u.arcsec**2 / u.um ) * u.arcsec**2
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
        min_area_px = min_aperture_area * u.pix**2 # closer to that seen in Fort Sumner and Alex+1999
        aperture_area_px = np.clip(aperture_area_px, min_area_px, None)
        logging.debug(f'aperture area clipped to {aperture_area_px}')

    sky_electrons_per_sec_per_px = sky_electrons_per_sec_per_sq_arcsec * (plate_scale_arcsec_per_px ** 2)

    source_electrons_per_px_per_exposure = source_electrons_per_sec * t / aperture_area_px * u.pix
    if source_electrons_per_px_per_exposure > starcam.sensor.full_well:
        logging.warning(
            f'SOURCE SATURATION: electron count from source mag {target_mag} exceeds sensor full well current, saturated!\n' +
            f'{source_electrons_per_px_per_exposure:.0f} > {starcam.sensor.full_well:.0f}\n' +
            f'sensor: {starcam.sensor.name}, lens: {starcam.lens.name}, filter: {starcam.filter.name}'
        )

    sky_electrons_per_px_per_exposure = sky_electrons_per_sec_per_px * t * u.pix
    if sky_electrons_per_px_per_exposure > starcam.sensor.full_well:
        logging.warning(
            f'SKY SATURATION: electron count from sky exceeds sensor full well current, saturated!\n' +
            f'{sky_electrons_per_px_per_exposure:.0f} > {starcam.sensor.full_well:.0f}\n' + 
            f'sensor: {starcam.sensor.name}, lens: {starcam.lens.name}, filter: {starcam.filter.name}'
        )

    if source_electrons_per_px_per_exposure + sky_electrons_per_px_per_exposure > starcam.sensor.full_well:
        logging.warning(
            f'SUM SATURATION: sum of sky and source counts exceeds sensor full well current, saturated!\n' +
            f'{source_electrons_per_px_per_exposure + sky_electrons_per_px_per_exposure:.0f} > {starcam.sensor.full_well:.0f}\n' + 
            f'sensor: {starcam.sensor.name}, lens: {starcam.lens.name}, filter: {starcam.filter.name}'
        )

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
        D,
        R,
        aperture_area_px.value,
        gain=1 # all values in electrons already
    )
    if return_components:
        return snr, (source_electrons_per_sec, sky_electrons_per_sec, aperture_area_px)
    return snr