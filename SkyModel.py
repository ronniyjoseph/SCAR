from __future__ import print_function
import numpy
import sys
from scipy import interpolate
from matplotlib import pyplot
import powerbox

def analytic_visibilities(baseline_table, frequencies, noise_param, sky_model,
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
        single_flux, single_l, single_m = flux_distribution(['single', sky_model[1], sky_model[2], sky_model[3]])

        all_flux = numpy.concatenate((single_flux, back_flux))
        all_l = numpy.concatenate((single_l, back_l))
        all_m = numpy.concatenate((single_m, back_m))
    else:
        sys.exit(str(sky_model[0]) + ": is not a correct input for " \
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
        model_visibilities[:, i] = point_source_visibility(all_flux, all_l,
                                                           all_m, baseline_table[:, 2, i], baseline_table[:, 3, i],
                                                           beam)

        ideal_visibilities[:, i] = model_visibilities[:, i] * baseline_table[:, 5, i] * \
                                   numpy.exp(1j * (baseline_table[:, 6, i]))

        amp_noise = numpy.random.normal(0, 1, n_measurements)
        phase_noise = numpy.random.normal(0, 1, n_measurements)

        obser_visibilities[:, i] = ideal_visibilities[:, i] + \
                                   noise_level * (amp_noise + 1j * phase_noise)

    return obser_visibilities, ideal_visibilities, model_visibilities


def numerical_visibilities(baseline_table, frequencies, noise_param, sky_model,
                       beam_param, seed):
    numpy.random.seed(seed)

    n_measurements = baseline_table.shape[0]
    n_frequencies = len(frequencies)
    # Select the sky model
    if sky_model[0] == 'background':
        all_flux, all_l, all_m = flux_distribution(['random', seed])
        point_source_list = numpy.stack((all_flux, all_l, all_m), axis=1)
    elif sky_model[0] == 'point':
        # extract point source coordinates from list
        all_flux, all_l, all_m = flux_distribution(['single', sky_model[1],
                                                    sky_model[2], sky_model[3]])
        point_source_list= numpy.stack((all_flux, all_l, all_m), axis=1)
    elif sky_model[0] == 'point_and_background':
        # extract point source coordinates from list
        back_flux, back_l, back_m = flux_distribution(['random', seed])
        single_flux, single_l, single_m = flux_distribution(['single', sky_model[1], sky_model[2], sky_model[3]])
        point_source_list = numpy.stack((numpy.concatenate((single_flux, back_flux)),
                                         numpy.concatenate((single_l, back_l)),
                                         numpy.concatenate((single_l, back_l))), axis=1)

    else:
        sys.exit(str(sky_model) + ": is not a correct input for " \
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
    sky_image, l_coordinates, m_coordinates = flux_list_to_sky_image(point_source_list, baseline_table)
    delta_l = numpy.diff(l_coordinates)

    if beam_param[0] == 'none':
        beam_attenuation = 1
    elif beam_param[0] == 'gaussian':
        beam_attenuation = beam_attenuator(sky_image, beam_param, frequencies)
    else:
        sys.exit(str(beam_param[0])+" is an invalid beam parameter, please change to 'none' or 'gaussian'")

    attenuated_image = sky_image*beam_attenuation
    #shift the zero point of the array to [0,0]
    shifted_image = numpy.fft.ifftshift(attenuated_image, axes=(0, 1))
    visibility_grid, uv_coordinates = powerbox.dft.fft(shifted_image, L=2., axes=(0, 1))
    normalised_visibilities = visibility_grid

    model_visibilities = uv_list_to_baseline_measurements(baseline_table, normalised_visibilities, uv_coordinates)
    ideal_visibilities = model_visibilities*baseline_table[:, 5, :]*numpy.exp(1j *baseline_table[:, 6, :])

    amp_noise = numpy.random.normal(0, 1, size =(n_measurements, n_frequencies))
    phase_noise = numpy.random.normal(0, 1, size=(n_measurements, n_frequencies))

    obs_visibilities = ideal_visibilities + noise_level * (amp_noise + 1j * phase_noise)

    return obs_visibilities, ideal_visibilities, model_visibilities


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
                      gamma2=2.5, S_low=400e-3, S_mid=1, S_high=5.):
    numpy.random.seed(seed)

    # Franzen et al. 2016
    # k1 = 6998, gamma1 = 1.54, k2=6998, gamma2=1.54
    # S_low = 0.1e-3, S_mid = 6.0e-3, S_high= 400e-3 Jy

    # Cath's parameters
    # k1=4100, gamma1 =1.59, k2=4100, gamma2 =2.5
    # S_low = 0.400e-3, S_mid = 1, S_high= 5 Jy

    if S_low > S_mid:
        norm = k2 * (S_high ** (1. - gamma2) - S_low ** (1. - gamma2)) / (1. - gamma2)
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


def flux_list_to_sky_image(point_source_list, baseline_table):
    #####################################
    #####################################
    # Assume the sky is flat
    #####################################

    #Converts list of sources into an image of the sky
    source_flux = point_source_list[:,0]
    source_l = point_source_list[:,1]
    source_m = point_source_list[:,2]

    #Find longest baseline to determine sky_image sampling, pick highest frequency for longest baseline
    max_u = numpy.max(numpy.abs(baseline_table[:, 2, -1]))
    max_v = numpy.max(numpy.abs(baseline_table[:, 3, -1]))
    max_b = max(max_u,max_v)

    #sky_resolutions
    min_l = 1./max_b
    delta_l = 0.1*min_l



    l_pixel_dimension = int(2./delta_l)
    if l_pixel_dimension % 2 == 0:
        l_pixel_dimension += 1
    n_frequencies = baseline_table.shape[2]

    #empty sky_image
    sky_image = numpy.zeros((l_pixel_dimension,l_pixel_dimension,n_frequencies))

    l_coordinates = numpy.linspace(-1, 1, l_pixel_dimension)



    l_shifts = numpy.diff(l_coordinates)/2.

    l_bin_edges = numpy.concatenate((numpy.array([l_coordinates[0] - l_shifts[0]]),
                                     l_coordinates[1:] - l_shifts,
                                     numpy.array([l_coordinates[-1] + l_shifts[-1]])))


    for frequency_index in range(n_frequencies):
        sky_image[:, :, frequency_index], l_bins, m_bins = numpy.histogram2d(source_l, source_m,
                                                           bins=(l_bin_edges, l_bin_edges),
                                                           weights=source_flux)

    #normalise skyimage for pixel size Jy/beam
    normalised_sky_image = sky_image/(2/l_pixel_dimension)**2.
    return sky_image, l_coordinates, l_coordinates


def uv_list_to_baseline_measurements(baseline_table, visibility_grid, uv_grid):
    n_frequencies = baseline_table.shape[2]
    n_measurements = baseline_table.shape[0]
    # #First of all convert the uv_grid to a bin_edges array
    # u_bin_centers = uv_grid[0]
    # v_bin_centers = uv_grid[1]
    #
    # u_shifts = numpy.diff(u_bin_centers)/2.
    # v_shifts = numpy.diff(v_bin_centers)/2.
    #
    # u_bin_edges = numpy.concatenate((numpy.array([u_bin_centers[0] - u_shifts[0]]),
    #                            u_bin_centers[1:] - u_shifts,
    #                            numpy.array([u_bin_centers[-1] + u_shifts[-1]])))
    # v_bin_edges = numpy.concatenate((numpy.array([v_bin_centers[0] - v_shifts[0]]),
    #                            v_bin_centers[1:] - v_shifts,
    #                            numpy.array([v_bin_centers[-1] + v_shifts[-1]])))

    #now we have the bin edges we can start binning our baseline table
    #Create an empty array to store our baseline measurements in
    visibilities = numpy.zeros((n_measurements, n_frequencies), dtype=complex)
    #print(len(uv_grid[0])
    #print(len(uv_grid[1])
    #print(visibility_grid.shape
    #print(baseline_table[:, 2, 0]
    for frequency_index in range(n_frequencies):
        visibility_data = visibility_grid[:, :, frequency_index]

        real_component = interpolate.RegularGridInterpolator((uv_grid[0], uv_grid[1]), numpy.real(visibility_data))
        imag_component = interpolate.RegularGridInterpolator((uv_grid[0], uv_grid[1]), numpy.imag(visibility_data))

        visibilities[:, frequency_index] = real_component(baseline_table[:, 2:4, frequency_index]) + \
                                           1j*imag_component(baseline_table[:, 2:4, frequency_index])

        #u_index = numpy.digitize(baseline_table[:, 2, frequency_index], bins=u_bin_edges)
        #v_index = numpy.digitize(baseline_table[:, 3, frequency_index], bins=v_bin_edges)

        #print("centers in u bins", u_bin_centers[u_index-1]
        #visibilities[:, frequency_index] = visibility_grid[u_index, v_index,frequency_index]

    return visibilities


def beam_attenuator(sky_image, beam_param, frequencies):
    l_coordinates = numpy.linspace(-1,1,sky_image.shape[0])
    m_coordinates = numpy.linspace(-1,1,sky_image.shape[1])

    l_mesh, m_mesh, frequency_mesh = numpy.meshgrid(l_coordinates,m_coordinates,frequencies,indexing="ij")
    width_l = beam_param[1]
    width_m = beam_param[2]
    if beam_param[0] == 'gaussian':
        beam_attenuation = numpy.exp(-0.5 * (l_mesh ** 2. / width_l ** 2. + m_mesh ** 2. / width_m ** 2.))
    elif beam_param[0] == 'none':
        beam_attenuation = numpy.zeros(l_mesh.shape) + 1
    else:
        sys.exit("Beam Parameter error: "+beam_param[0] +" should be gaussian or none")
    #beam_attenuation = numpy.tile(beam_image,(2,sky_image.shape[2]))
    return beam_attenuation