import numpy as np
from astropy import units as u
from astropy import constants as c
import astropy.coordinates as coord
from astropy.stats import signal_to_noise_oir_ccd
from astroquery.vizier import Vizier
from astroquery.xmatch import XMatch

from scipy.interpolate import PchipInterpolator


class InterpolatableTable(object):
    def __init__(self, x, y):
        assert x.shape == y.shape
        self.x = x
        self.y = y

    def interp(self, x_new):
        pchip_interp = PchipInterpolator(self.x, self.y)
        return pchip_interp(x_new)


class Sensor(object):
    def __init__(self, shape, px_size, dark_current, read_noise, bits, qe_table=None, name=None, cost=0):
        self.shape = shape * u.pix
        self.px_size = px_size
        self.dark_current = dark_current
        self.read_noise = read_noise

        self.megapixels = self.shape[0] * self.shape[1] / 1e6 / u.pix**2
        self.sensor_size = (self.shape * self.px_size / u.pix).to(u.mm)
        self.bits = bits
        self.frame_storage_size = bits * self.megapixels * 1e6

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
    return np.ones_like(lambd.value) * 0.98


def get_filter_transmission(lambd, center=650*u.nm, center2=None, width=7*u.nm, max_transmission=0.9):
    # scaled and shifted sigmoid
    scale = 1. / width
    if center2 is None:
        return max_transmission * (1. / (1. + np.exp(-scale * (lambd - center))))
    else: # Hack to test narrowband filters
        cut0 = max_transmission * (1. / (1. + np.exp(-scale * (lambd - center))))
        cut1 = max_transmission - max_transmission * (1. / (1. + np.exp(-scale * (lambd - center2))))
        return cut0 * cut1 / (np.max(cut0*cut1))


def get_sky_brightness(lambd):
    # Alexander 1999, Fig. 4
    # MODTRAN, 35km, 30deg SZA
    # note: sky brightness may actually go UP past 1200 nm! water, etc.
    plot_lambd = np.array([400, 500, 600, 700, 800, 900, 1000, 1100, 1200]) * u.nm
    plot_val = 1e-5 * np.array([10., 5., 2.2, 1., 0.5, 0.2, 0.1, 0.0, 0.0]) * u.W / u.cm**2 / u.sr / u.um
    interp = PchipInterpolator(plot_lambd, plot_val)
    return interp(lambd) * plot_val.unit


def calc_limiting_mag(snr_limit, mags, snrs):
    snr_table = InterpolatableTable(snrs, mags)
    return snr_table.interp(snr_limit)


def get_tycho_stars(coord, width, height, mag_limit=9, nlimit=1000):
    query = Vizier(
        columns=['RAmdeg', 'DEmdeg', 'BTmag', 'VTmag'],
        column_filters={'VTmag' : f'<{mag_limit}'},
        row_limit=10000
    )
    table = query.query_region(coord, width=width, height=height, catalog='I/259/tyc2')[0]
    table.sort('VTmag')
    return table[:nlimit]


def get_median_Teff(coord, width, height, match_radius=1*u.arcsec, return_array=False):
    tycho_stars = get_tycho_stars(coord, width, height, mag_limit=9, nlimit=1000)
    # Get a list of GAIA Teffs by cross-referencing the given ra/dec
    gaia_dr3 = 'vizier:I/355/paramp'
    table = XMatch.query(
        cat1=tycho_stars,
        colRA1='RAmdeg',
        colDec1='DEmdeg',
        cat2=gaia_dr3,
        max_distance=match_radius
    )
    nonzero = table['Teff'] > 0
    Teffs = table['Teff'][nonzero]
    if return_array:
        return np.median(Teffs) * u.K, Teffs * u.K
    return np.median(Teffs) * u.K


def get_model_sed(lambd, Teff):
    # it's the Planck function
    def b(lambd, T):
        # wien
        num = 2. * c.h * c.c ** 2.
        expn = c.h * c.c / (lambd * c.k_B * T)
        return num / ((lambd ** 5.) * (np.exp(expn) - 1.)) / u.sr
    sed = b(lambd, Teff).to(u.W / u.cm**2 / u.um / u.sr) * u.sr
    return sed / sed.max()


def get_zero_point_flux(lambd, coord, width, height, filter: Filter):
    # Bootstrap off of an existing survey zero-point flux:
    # TYCHO2 V: http://svo2.cab.inta-csic.es/svo/theory/fps/index.php?mode=browse&gname=TYCHO&asttype=
    # Vega system, non-Bessel calibration
    # calibration reference: https://ui.adsabs.harvard.edu/abs/1995A&A...304..110G/abstract
    F0 = (3.99504e-9 * u.erg / u.s / u.cm**2 / u.angstrom).to(u.W / u.cm**2 / u.um)
    lambd_dat, tau = np.genfromtxt('./TYCHO_TYCHO.V.dat', dtype=float, unpack=True)
    lambd_dat = (lambd_dat * u.angstrom).to(u.nm)
    tycho_v = Filter(F0, tau_table=InterpolatableTable(lambd_dat, tau), name='Tycho V')

    # Get median Teff in field
    Teff = get_median_Teff(coord, width, height)
    # Get a normalized SED for that Teff
    sed = get_model_sed(lambd_dat, Teff)
    # Scale our SED to give the same average flux as Vega over the V filter
    dl = np.diff(lambd_dat).mean()
    sed_int = np.trapz(sed * tycho_v.tau(lambd_dat), dx=dl) / np.trapz(tycho_v.tau(lambd_dat), dx=dl)
    norm = F0 / sed_int
    rescaled_sed = get_model_sed(lambd, Teff) * norm
    # The average flux of our new SED in our arbitrary filter is our zero-point flux
    dl = np.diff(lambd).mean()
    F0_new = np.trapz(rescaled_sed * filter.tau(lambd), dx=dl) / np.trapz(filter.tau(lambd), dx=dl)
    return F0_new


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
    return (1. / (c.h * c.c)) * A_tel * np.trapz(lambd * eta * tau * flux, dx=lambd.diff().mean())


def simple_snr_spectral(t, lambd, target_mag, starcam: StarCamera, min_aperture_area=4):
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
        For optics with diffraction-limited resolution finer than the plate scale
        (undersampled), use this number of pixels as the minimum number of pixels
        for calculating the sky noise contribution. 4 is the default because a star
        that falls at the corner of a pixel will have its light spread evenly across a
        4 pixel area.

    Returns
    -------
    float
        Signal-to-noise ratio in a single exposure
    '''
    D = starcam.sensor.dark_current
    R = starcam.sensor.read_noise
    plate_scale_arcsec_per_px = starcam.plate_scale.mean()
    psf_diam = starcam.lens.fwhm(lambd).mean()

    # Source signal
    flux = starcam.filter.zero_point_flux * (10. ** (-0.4 * target_mag))
    source_electrons_per_sec = electrons_per_sec_spectral(
        starcam.lens.tau(lambd) * starcam.filter.tau(lambd), # total optical transmission
        starcam.sensor.qe(lambd),
        starcam.lens.area,
        lambd,
        flux
    ).decompose()

    # Model gives model gives W/cm^2/sr/um, want sky signal in 1 arcsec^2:
    sky_flux = get_sky_brightness(lambd).to(u.W / u.cm**2 / u.arcsec**2 / u.um ) * u.arcsec**2
    sky_electrons_per_sec_per_sq_arcsec = electrons_per_sec_spectral(
        starcam.lens.tau(lambd) * starcam.filter.tau(lambd), # total optical transmission
        starcam.sensor.qe(lambd),
        starcam.lens.area,
        lambd,
        sky_flux
    ).decompose() / u.arcsec**2

    # fake aperture photometry: all source photons end up inside the aperture

    aperture_area_px = np.pi * (psf_diam.to(u.arcsec) / plate_scale_arcsec_per_px.to(u.arcsec / u.pix) / 2.)**2
    # print('aperture area px', aperture_area_px)
    if min_aperture_area > 0:
        # the aperture is matched to the greater of the two: PSF or the minimum.
        # if seeing is to be considered, it shall be considered in the psf diameter.
        # min_area_px = 1 # no optics/ice/dust scattering/smearing
        min_area_px = min_aperture_area * u.pix**2 # closer to that seen in Fort Sumner and Alex+1999
        aperture_area_px = np.clip(aperture_area_px, min_area_px, None)
        # print('aperture area clipped to', aperture_area_px)

    sky_electrons_per_sec_per_px = sky_electrons_per_sec_per_sq_arcsec * (plate_scale_arcsec_per_px ** 2)
    
    sky_electrons_per_sec = sky_electrons_per_sec_per_px * aperture_area_px
    # print('source e-:', source_electrons_per_sec)
    # print('psf diameter', psf_diam.to(u.arcsec))
    # print('sky e-/s/px', sky_electrons_per_sec_per_px)
    # print('sky e-/s', sky_electrons_per_sec)

    snr = signal_to_noise_oir_ccd(
        t,
        source_electrons_per_sec,
        sky_electrons_per_sec,
        D,
        R,
        aperture_area_px.value,
        gain=1 # all values in electrons already
    )
    return snr