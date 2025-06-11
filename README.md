# starcam-sim

Performance simulations for daytime star cameras for subortibal missions.

## Design

This calculator is organized as a "composable system": Component models for filters, lenses, and sensors can be combined into a Star Camera object. This object has awareness of child properties (filter transmission curves, focal length, sensor size) and those properties that are derived from the composition (plate scale, field of view, total response curve).

This is a powerful tool for trade studies, where combining simple models of the sky and stars can give performance indications like SNR of individual stars in the field, number of detected stars, and ancillary data like cost and storage footprints.

## Assumptions

### Stellar Modeling

* No atmospheric exctinction
    * The use case is a balloon-borne or space mission
* Relevant catalog for performance is Tycho-2
    * astrometry.net is based on Tycho-2 but supplementary catalogs are available.
* Stars are well-approximated by blackbodies of a given effective temperature (T_eff)
* The magnitude limits in a given FoV are well-approximated by assuming all stars have the median T_eff of stars in the FoV
    * This is a simplifying assumption, and predictive performance would improve if each star's T_eff were used

### Atmosphere Modeling

* Atmospheric emission is smooth
    * Does not account for emission or absorption lines
* Absolute value of emission spectrum is approximate
    * Variations in altitude or sun-relative angle not accounted for, except to take the worst-case values from Alexander+ 1999 MODTRAN model for a suborbital flight ~35 km

### Instrument Modeling

* Photographic lenses are not diffraction-limited
    * The degree to which the PSF is degraded varies. As a first-order approximation, for the SNR calculation of background counts, we multiply the diffraction-limited PSF by a factor of 13, which forces the PSF to agree with the measured value for a well-corrected lens, a Sigma 85mm f/1.4 DG HSM ART.
* Sensor noise is dominated by sky background noise
    * We use the best publically available values for read noise and dark current, and incorporate them into the SNR calculations.
    * We assume the impact of inaccurate dark current or read noise figures is negligible, and so the results are valid even if accurate dark current or read noise figures are unavailable.
        * Modern CMOS read noise is small
        * Dark current, even for uncooled sensors, is negligible in the short exposure limit, where most attitude determination cameras operate

## Contributing

If you'd like to contribute to this repo, simply open a pull request and I'll review it.

* New sensor models?
* New lens models?
* New filters?
* New atmosphere models?
