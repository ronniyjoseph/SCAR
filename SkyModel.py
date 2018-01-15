import numpy
import sys


def CreateVisibilities(baseline_table, frequencies, noise_param, sky_model,
                       beam, seed):
    # Select the sky model
    if sky_model[0] == 'background':
        all_flux, all_l, all_m = flux_distribution(['random', seed])
    elif sky_model[0] == 'point':
        # extract point source coordinates from list
        all_flux, all_l, all_m = flux_distribution(['single', sky_model[1],
													sky_model[2], sky_model[3]])
    elif sky_model[0] == 'point_and_background':
        # extract point source coordinates from list
        back_flux, back_l, back_m = flux_distribution(['random', seed])
        single_flux, single_l, single_m = flux_distribution(['single', \
                                                             sky_model[1], sky_model[2], sky_model[3]])
        all_flux = numpy.concatenate((single_flux, back_flux))
        all_l = numpy.concatenate((single_l, back_l))
        all_m = numpy.concatenate((single_m, back_m))
    else:
        sys.exit(str(noise) + ": is not a correct input for " \
                              "create_visibilities. Please adjust skymodel parameter")

    if noise_param[0] == 'source':
        noise_level = 0.1 * max(all_flux)
    elif noise_param[0] == 'SEFD':
        SEFD = noise_param[1]
        bandwidth = noise_param[2]
        t_integration = noise_param[3]
        noise_level = sky_noise(SEFD, bandwidth, t_integration)
    elif noise_param[0] == False:
        noise_level = 0.
    else:
        sys.exit(str(noise_param[0]) + ": is not a correct input for " \
                                       "create_mock_observations. True or False, please for " \
                                       "the noise variable")

    # Calculate the ideal measured amplitudes for these sources at different
    # frequencies
    n_measurements = baseline_table.shape[0]

    model_visibilities = numpy.zeros((n_measurements, len(frequencies)), dtype=complex)
    obser_visibilities = numpy.zeros((n_measurements, len(frequencies)), dtype=complex)
    ideal_visibilities = numpy.zeros((n_measurements, len(frequencies)), dtype=complex)

    for i in range(len(frequencies)):
        model_visibilities[:, i] = point_source_visibility(all_flux, all_l, \
                                                           all_m, baseline_table[:, 2, i], baseline_table[:, 3, i],
                                                           beam)

        ideal_visibilities[:, i] = model_visibilities[:, i] * baseline_table[:, 5, i] * \
                                   numpy.exp(1j * (baseline_table[:, 6, i]))

        amp_noise = numpy.random.normal(0, 1, n_measurements)
        phase_noise = numpy.random.normal(0, 1, n_measurements)

        obser_visibilities[:, i] = ideal_visibilities[:, i] + \
                                   noise_level * (amp_noise + 1j * phase_noise)

    return obser_visibilities, ideal_visibilities, model_visibilities


def flux_distribution(model):
    if model[0] == 'random':
        # Random background sky
        all_flux = source_population(model[1])  # S_low=400e-3,S_high=5)
        # all_l = numpy.random.uniform(-1,1,len(all_flux))
        # all_m = numpy.random.uniform(-1,1,len(all_flux))
        numpy.random.seed(model[1])
        all_r = numpy.sqrt(numpy.random.uniform(0, 1, len(all_flux)))
        all_phi = numpy.random.uniform(0, 2. * numpy.pi, len(all_flux))
        all_l = all_r * numpy.cos(all_phi)
        all_m = all_r * numpy.sin(all_phi)
    elif model[0] == 'single':
        all_flux = numpy.array([model[1]])
        all_l = numpy.array([model[2]])
        all_m = numpy.array([model[3]])
    return all_flux, all_l, all_m


def source_population(seed, k1=4100, gamma1=1.59, k2=4100, \
                      gamma2=2.5, S_low=400.e-3, S_mid=1, S_high=5.):
    # Franzen et al. 2016
    # k1 = 6998, gamma1 = 1.54, k2=6998, gamma2=1.54
    # S_low = 0.1e-3, S_mid = 6.0e-3, S_high= 400e-3 Jy

    # Cath's parameters
    # k1=4100, gamma1 =1.59, k2=4100, gamma2 =2.5
    # S_low = 0.400e-3, S_mid = 1, S_high= 5 Jy

    if S_low > S_mid:
        norm = k2 * (S_high ** (1. - gamma2) - S_low ** (1. - gamma2)) / (1. - gamma2)
        numpy.random.seed(seed)
        n_sources = numpy.random.poisson(norm * 2. * numpy.pi)
        # generate uniform distribution
        uniform_distr = numpy.random.uniform(size=n_sources)
        # initialize empty array for source fluxes
        source_fluxes = numpy.zeros(n_sources)
        source_fluxes = \
            (uniform_distr * norm * (1. - gamma2) / k2 +
             S_low ** (1. - gamma2)) ** (1. / (1. - gamma2))
    else:
        # normalisation
        norm = k1 * (S_mid ** (1. - gamma1) - S_low ** (1. - gamma1)) / (1. - gamma1) + \
               k2 * (S_high ** (1. - gamma2) - S_mid ** (1. - gamma2)) / (1. - gamma2)
        # transition between the one power law to the other
        mid_fraction = k1 / (1. - gamma1) * (S_mid ** (1. - gamma1) - S_low ** (1. - gamma1)) / norm
        numpy.random.seed(seed)
        n_sources = numpy.random.poisson(norm * 2. * numpy.pi)

        #########################
        # n_sources = 1e5
        #########################

        # generate uniform distribution
        uniform_distr = numpy.random.uniform(size=n_sources)
        # initialize empty array for source fluxes
        source_fluxes = numpy.zeros(n_sources)

        source_fluxes[uniform_distr < mid_fraction] = \
            (uniform_distr[uniform_distr < mid_fraction] * norm * (1. - gamma1) / k1 +
             S_low ** (1. - gamma1)) ** (1. / (1. - gamma1))

        source_fluxes[uniform_distr >= mid_fraction] = \
            ((uniform_distr[uniform_distr >= mid_fraction] - mid_fraction) * norm * (1. - gamma2) / k2 +
             S_mid ** (1. - gamma2)) ** (1. / (1. - gamma2))
    return source_fluxes


def sky_noise(SEFD=20e3, B=40e3, t=120.):
    """ Calculates the sky noise as a function of wavelength, Bandwith
	B and integration time t. Standard values are for the MWA EoR experiment
	
	wavelength	:	 wavelength of interest in meters
	B	:	Bandwidth in Hz
	t	:	integration time in seconds
	SEFD	: System equivalent Flux Density for MWA = 20 10^3 Jy
	"""
    noise = SEFD / numpy.sqrt(B * t)
    return noise


def point_source_visibility(flux, l, m, u, v, beam):
    if len(flux) != len(l):
        sys.exit("length flux, l,m is unequal")
    source_visibilities = numpy.zeros(len(u), dtype=complex)
    for i in range(len(flux)):
        if beam[0] == 'none':
            source_visibilities += flux[i] * \
                                   numpy.exp(-2. * numpy.pi * 1j * (u * l[i] + v * m[i]))
        elif beam[0] == 'gaussian':
            width_l = beam[1]
            width_m = beam[2]
            point_source = flux[i] * \
                           numpy.exp(-2. * numpy.pi * 1j * (u * l[i] + v * m[i]))
            attenuation = numpy.exp(-0.5 * (l[i] ** 2. / width_l ** 2. + m[i] ** 2. / width_m ** 2.))
            source_visibilities += point_source * attenuation
        else:
            sys.exit(beam[0] + " is an invalid beam parameter. Please " + \
                     "choose from 'none' or 'gaussian'")
    return source_visibilities
