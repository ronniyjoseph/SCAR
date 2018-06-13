import numpy
import os
import sys
import time
from matplotlib import pyplot

from RadioTelescope import antenna_gain_creator
from RadioTelescope import baseline_converter
from RadioTelescope import redundant_baseline_finder
from RadioTelescope import xyz_position_creator
from RadioTelescope import antenna_table_loader
from GeneralTools import unique_value_finder
from GeneralTools import position_finder
from GeneralTools import solution_averager
from GeneralTools import FourD_solution_averager
from GeneralTools import visibility_histogram_plotter
from GeneralTools import solution_histogram_plotter
from GeneralTools import TrueSolutions_Organizer
from GeneralTools import save_to_hdf5
from SkyModel import numerical_visibilities
from SkyModel import analytic_visibilities
from RedundantCalibration import Redundant_Calibrator
from RedundantCalibration import LogcalMatrixPopulator

"""Simulate Calibration with Array Redundancy
"""


def Moving_Source(telescope_param, offset_param, calibration_channel, noise_param, direction,
                  sky_steps, input_iterations, sky_param, beam_param, calibration_scheme, save_to_disk, hist_movie):
    starttime = time.time()

    if not os.path.exists(save_to_disk[1]):
        print ""
        print "!!!Warning: Creating output folder at output destination!"
        print save_to_disk[1]
        os.makedirs(save_to_disk[1])



    if telescope_param[0] == 'square' \
            or telescope_param[0] == 'hex' \
            or telescope_param[0] == 'doublehex' \
            or telescope_param[0] == 'doublesquare' \
            or telescope_param[0] == 'linear':
        xyz_positions = xyz_position_creator(telescope_param)
    else:
        xyz_positions = antenna_table_loader(telescope_param[0])

    if offset_param[0] == True:
        if offset_param[2] == 'x':
            print "offsetting tile", offset_param[1], "by", offset_param[3], "meters"
            xyz_positions[offset_param[1], 1] += offset_param[3]
        elif offset_param[2] == 'y':
            xyz_positions[offset_param[1], 2] += offset_param[3]

    frequency_range = numpy.array(calibration_channel)
    gain_table = antenna_gain_creator(xyz_positions, frequency_range)
    baseline_table = baseline_converter(xyz_positions, gain_table,
                                        frequency_range)

    ###################################################################
    #								intra sub array selecter
    ###################################################################
    # ~ if telescope_param[0] == 'doublesquare' or telescope_param[0] == 'doublehex' :
    # ~ hex1_boolean = (baseline_table[:,0] < 2000)
    # ~ hex2_boolean = (baseline_table[:,1] < 2000)
    # ~ intra_hex_index  =  numpy.equal(hex1_boolean[:,0], hex2_boolean[:,0])
    # ~ baseline_table = baseline_table[intra_hex_index,:]

    print "Simulating the Calibration of Arrays with Redundancy (SCAR)"
    print "Changing source position for fixed input parameters"

    # Find the redundant tiles
    red_baseline_table = redundant_baseline_finder(baseline_table, 'ALL', verbose = True)
    # Calculate the solving matrices (only needs to be once)
    amp_matrix, phase_matrix, red_tiles, red_groups = LogcalMatrixPopulator(
        red_baseline_table, xyz_positions)

    # Double check input parameters whether they suit the requirements if not change them.
    iterations, sky_steps, type_sim = check_noise_and_sky_parameters(noise_param, sky_param, sky_steps,
                                                                     input_iterations)

    # Create empty 3D table to store calibration results as a function
    # of realization and sky positions
    n_measurements = red_baseline_table.shape[0]
    n_tiles = len(red_tiles)
    n_groups = len(red_groups)

    noisy_amp_solutions = numpy.zeros((n_tiles + n_groups,
                                       sky_steps, iterations))
    noisy_phase_solutions = numpy.zeros((n_tiles + n_groups,
                                         sky_steps, iterations))

    ideal_amp_solutions = numpy.zeros((n_tiles + n_groups,
                                       sky_steps, iterations))
    ideal_phase_solutions = numpy.zeros((n_tiles + n_groups,
                                         sky_steps, iterations))

    if hist_movie[0]:
        if noise_param[0]:
            amp_obs = numpy.zeros((n_measurements,
                                   sky_steps, iterations))
            phase_obs = numpy.zeros((n_measurements,
                                     sky_steps, iterations))
            amp_mod = numpy.zeros((n_measurements,
                                   sky_steps, iterations))
            phase_mod = numpy.zeros((n_measurements,
                                     sky_steps, iterations))

    random_seeds = numpy.arange(iterations)
    sky_coords = numpy.linspace(-1, 1, sky_steps)

    print ""
    print "Simulating redundant calibration with a %s %s sky" % (type_sim, sky_param[0])


    file = open(save_to_disk[1] + "simulation_parameter.log", "w")
    file.write("Standard Redundant Calibration Simulation" + "\n")
    file.write("Telescope Parameters: " + str(telescope_param) + "\n")
    file.write("Telescope Offsets: " + str(offset_param) + "\n")
    file.write("Calibration Channel: " + str(frequency_range / 1e6) + "MHz \n")
    file.write("Calibration Scheme: " + str(calibration_scheme) + "\n")
    file.write("Iterations: " + str(iterations) + "\n")
    file.write("Noise Parameters: " + str(noise_param) + "\n")
    file.write("Source Direction: " + direction + "\n")
    file.write("Sky Steps: " + str(sky_steps) + "\n")
    file.write("Sky Model: " + str(sky_param) + "\n")
    file.write("Iterations: " + str(iterations) + "\n")
    file.write("Beam Parameters: " + str(beam_param) + "\n")
    file.write("Save Parameters: " + str(save_to_disk) + "\n")
    file.close()


    for j in range(iterations):
        if numpy.mod(j, 100) == 0:
            print "Realization", j
        # seed = numpy.random.randint(1000)
        seed = random_seeds[j]
        if sky_param[0] == "background" or sky_param[0] == 'point_and_background':
            # Create the visibilities for the static background sky
            sky_model = ['background']
            obs_visibilities, ideal_visibilities, model_visibilities = \
                numerical_visibilities(red_baseline_table, frequency_range,
                                   [False], sky_model, beam_param, seed)

        for i in range(sky_steps):
            if direction == 'l':
                l = sky_coords[i]
                m = 0

            elif direction == 'm':
                l = 0
                m = sky_coords[i]

            # add a point source (with noise) to background
            if sky_param[0] == 'point_and_background':
                sky_model = ['point', sky_param[1], l, m]
                if noise_param[0] and len(noise_param) == 4:
                    noise_param[0] = 'SEFD'
                elif noise_param[0] and len(noise_param) == 1:
                    noise_param[0] = 'source'
                point_obs_visibilities, point_ideal_visibilities, point_model_visibilities = \
                    numerical_visibilities(red_baseline_table, frequency_range
                                       ,noise_param, sky_model, beam_param, seed)

                obs_visibilities += point_obs_visibilities
                ideal_visibilities += point_ideal_visibilities
                model_visibilities += point_model_visibilities

            # add noise to the background sky
            elif sky_param[0] == 'background':
                # Setting the skymodel point source to 0, so just add noise
                sky_model = ['point', 0, 0, 0]
                if noise_param[0] and len(noise_param) == 4:
                    noise_param[0] = 'SEFD'
                elif noise_param[0] and len(noise_param) == 1:
                    noise_param[0] = 'source'
                point_obs_visibilities, point_ideal_visibilities, point_model_visibilities = \
                    numerical_visibilities(red_baseline_table, frequency_range,
                                       noise_param, sky_model, beam_param, seed)

                obs_visibilities += point_obs_visibilities
                ideal_visibilities += point_ideal_visibilities
                model_visibilities += point_model_visibilities

            # Create point source data in the absence of background sky data
            elif sky_param[0] == 'point':
                sky_model = ['point', sky_param[1], l, m]
                noise_param[0] = 'source'
                obs_visibilities, ideal_visibilities, model_visibilities = \
                    numerical_visibilities(red_baseline_table, frequency_range
                                       , noise_param, sky_model, beam_param, seed)

            if calibration_scheme == 'lincal':
                true_solutions = TrueSolutions_Organizer(gain_table,
                                                         model_visibilities, red_baseline_table, red_tiles, red_groups)
                calibration_param = ['lincal', true_solutions]
            elif calibration_scheme == 'logcal' or calibration_scheme == 'full':
                calibration_param = [calibration_scheme]
            else:
                sys.exit("You've chosen an invalid calibration parameter")

            # Use the model data  to solve for the antenna gains
            ideal_amp_data, ideal_phase_data = Redundant_Calibrator(
                amp_matrix, phase_matrix, ideal_visibilities,
                red_baseline_table, red_tiles, red_groups, calibration_param)

            ideal_amp_solutions[:, i, j] = ideal_amp_data
            ideal_phase_solutions[:, i, j] = ideal_phase_data

            # Use the noisy data  to solve for the antenna gains
            noisy_amp_data, noisy_phase_data = Redundant_Calibrator(
                amp_matrix, phase_matrix, obs_visibilities,
                red_baseline_table, red_tiles, red_groups, calibration_param)

            noisy_amp_solutions[:, i, j] = noisy_amp_data
            noisy_phase_solutions[:, i, j] = noisy_phase_data

            if hist_movie[0]:
                if noise_param[0]:
                    amp_obs[:, i, j] = numpy.absolute(obs_visibilities[:, 0])
                    phase_obs[:, i, j] = numpy.angle(obs_visibilities[:, 0])
                    amp_mod[:, i, j] = numpy.absolute(model_visibilities[:, 0])
                    phase_mod[:, i, j] = numpy.angle(model_visibilities[:, 0])

            # remove the point source
            if sky_param[0] == 'point_and_background':
                obs_visibilities -= point_obs_visibilities
                ideal_visibilities -= point_ideal_visibilities
                model_visibilities -= point_model_visibilities

    parameters = numpy.concatenate((red_tiles, red_groups))
    axesdata = [parameters, sky_coords, random_seeds]
    axeslabels = ['parameters', 'l_coordinates', 'iteration']


    save_to_hdf5(save_to_disk[1], "ideal_amp_solutions", ideal_amp_solutions,
                 axesdata, axeslabels)
    save_to_hdf5(save_to_disk[1], "ideal_phase_solutions", ideal_phase_solutions,
                 axesdata, axeslabels)

    save_to_hdf5(save_to_disk[1], "noisy_amp_solutions", noisy_amp_solutions,
                 axesdata, axeslabels)
    save_to_hdf5(save_to_disk[1], "noisy_phase_solutions", noisy_phase_solutions,
                 axesdata, axeslabels)

    # Calculate run time
    endtime = time.time()
    runtime = endtime - starttime

    # Save input parameters to log file
    file = open(save_to_disk[1] + "simulation_parameter.log", "a")
    file.write("Runtime: " + str(runtime) + "\n")
    file.close()

    print "Runtime", runtime
    return


def MuChSource_Mover(n_channels, telescope_param, calibration_channel, noise_param, direction,
                     sky_steps, sky_param, beam_param, save_to_disk):
    # Track how long it's taking
    starttime = time.time()
    print "Simulating the Calibration of Arrays with Redundancy (SCAR)"
    print "Changing source position and position offsets with a multi channel implementation"
    xyz_positions = xyz_position_creator(telescope_param)
    channel_size = noise_param[2]

    # calculate the frequencies of the adjecent channels
    frequency_range = numpy.arange(calibration_channel - n_channels * channel_size,
                                   calibration_channel + (n_channels + 1) * channel_size, channel_size)

    gain_table = antenna_gain_creator(xyz_positions, frequency_range)
    baseline_table = baseline_converter(xyz_positions, gain_table,
                                        frequency_range)

    print ""
    # Find the redundant tiles
    red_baseline_table = redundant_baseline_finder(baseline_table, 'ALL')

    # Calculate the solving matrices (only needs to be calculated once
    # and includes all frequency channel measurements
    amp_matrix, phase_matrix, red_tiles, red_groups = \
        MuChMatrixPopulator(red_baseline_table, xyz_positions)

    type_sim = ""

    if noise_param[0]:
        iterations = 1001
        type_sim += " noisy "
    else:
        iterations = 1
        type_sim += " ideal "

    if sky_param[0] == 'background' or sky_param[0] == 'point_and_background':
        iterations = 1001

    # Create empty 3D table to store calibration results as a function
    # of realization and sky positions
    n_measurements = red_baseline_table.shape[0]
    n_frequencies = red_baseline_table.shape[2]
    middle_index = (n_frequencies + 1) / 2 - 1
    n_tiles = len(red_tiles)
    n_groups = len(red_groups)

    calibration_frequencies = numpy.delete(frequency_range, middle_index)
    calibration_baselines = numpy.delete(red_baseline_table, middle_index, axis=2)

    if sky_param[0] == "point" or sky_param[0] == 'point_and_background':
        if sky_param[0] == "point":
            type_sky = "%s Jy point source sky" % str(sky_param[1])
        elif sky_param[0] == 'point_and_background':
            type_sky = "%s Jy point source and background sky" % str(sky_param[1])

    elif sky_param[0] == 'background':
        sky_steps = 1
        type_sky = 'background sky'
    else:
        sys.exit(sky_param[0] + " is an invalid sky model parameter. Please " + \
                 "choose from 'point' or 'background' or 'point_and_background'")

    noisy_amp_solutions = numpy.zeros((n_tiles + n_groups,
                                       sky_steps, iterations))
    noisy_phase_solutions = numpy.zeros((n_tiles + n_groups,
                                         sky_steps, iterations))

    ideal_amp_solutions = numpy.zeros((n_tiles + n_groups,
                                       sky_steps, iterations))
    ideal_phase_solutions = numpy.zeros((n_tiles + n_groups,
                                         sky_steps, iterations))

    sky_coords = numpy.linspace(-1, 1, sky_steps)
    random_seeds = numpy.arange(iterations)
    print ""
    print "Simulating Multi-Channel Redundant Calibration with a" + type_sim + type_sky
    print "%s beam and %d frequency channels" % (beam_param[0], n_channels)
    for j in range(iterations):
        # seed = numpy.random.randint(1000)
        seed = random_seeds[j]
        if sky_param[0] == "background" or sky_param[0] == 'point_and_background':
            # Create the visibilities for the static background sky
            sky_model = ['background']
            obs_visibilities, ideal_visibilities, model_visibilities = \
                CreateVisibilities(calibration_baselines,
                                   calibration_frequencies, [False], sky_model, beam_param,
                                   seed)

        for i in range(sky_steps):
            if direction == 'l':
                l = numpy.array([sky_coords[i]])
                m = numpy.array([0])

            elif direction == 'm':
                l = numpy.array([0])
                m = numpy.array([sky_coords[i]])

            # add a point source (with noise) to background
            if sky_param[0] == 'point_and_background':
                sky_model = ['point', sky_param[1], l, m]
                if noise_param[0] and len(noise_param) == 4:
                    noise_param[0] = 'SEFD'
                elif noise_param[0] and len(noise_param) == 1:
                    noise_param[0] = 'source'
                point_obs_visibilities, point_ideal_visibilities, \
                point_model_visibilities = \
                    CreateVisibilities(calibration_baselines,
                                       calibration_frequencies, noise_param, sky_model, beam_param,
                                       seed)

                obs_visibilities += point_obs_visibilities
                ideal_visibilities += point_ideal_visibilities
                model_visibilities += point_model_visibilities
            # add noise to the background sky

            elif sky_param[0] == 'background':
                # Setting the skymodel point source to 0, so just add noise
                sky_model = ['point', 0, 0, 0]
                if noise_param[0] and len(noise_param) == 4:
                    noise_param[0] = 'SEFD'
                elif noise_param[0] and len(noise_param) == 1:
                    noise_param[0] = 'source'
                point_obs_visibilities, point_ideal_visibilities, \
                point_model_visibilities = \
                    CreateVisibilities(
                        calibration_baselines, calibration_frequencies,
                        noise_param, sky_model, beam_param, seed)

                obs_visibilities += point_obs_visibilities
                ideal_visibilities += point_ideal_visibilities
                model_visibilities += point_model_visibilities

            # Create point source data in the absence of background sky data
            elif sky_param[0] == 'point':
                sky_model = ['point', sky_param[1], l, m]
                noise_param[0] = 'SEFD'

                obs_visibilities, ideal_visibilities, \
                model_visibilities = \
                    CreateVisibilities(calibration_baselines,
                                       calibration_frequencies, noise_param, sky_model, beam_param,
                                       seed)

            # Use the model data  to solve for the antenna gains
            ideal_amp_data, ideal_phase_data = MuChRedCalibrator(amp_matrix,
                                                                 phase_matrix, ideal_visibilities)

            ideal_amp_solutions[:, i, j] = ideal_amp_data
            ideal_phase_solutions[:, i, j] = ideal_phase_data

            # Use the noisy data  to solve for the antenna gains
            noisy_amp_data, noisy_phase_data = MuChRedCalibrator(amp_matrix,
                                                                 phase_matrix, obs_visibilities)

            noisy_amp_solutions[:, i, j] = noisy_amp_data
            noisy_phase_solutions[:, i, j] = noisy_phase_data

            # remove the point source
            if sky_param[0] == 'point_and_background':
                obs_visibilities -= point_obs_visibilities
                ideal_visibilities -= point_ideal_visibilities
                model_visibilities -= point_model_visibilities

    noisy_amp_info, noisy_phase_info = solution_averager(
        noisy_amp_solutions, noisy_phase_solutions, red_tiles, red_groups,
        sky_coords, save_to_disk, direction, [True])
    ideal_amp_info, ideal_phase_info = solution_averager(
        ideal_amp_solutions, ideal_phase_solutions, red_tiles, red_groups,
        sky_coords, save_to_disk, direction, [False])
    if save_to_disk[0]:
        file = open(save_to_disk[1] + "simulation_parameter.log", "w")
        file.write("Multi-Channel Redundant Calibration Simulation\n")
        file.write("Telescope Parameters: " + str(telescope_param) + "\n")
        file.write("Number Channels: " + str(n_channels) + "\n")
        file.write("Calibration Channel: " + str(calibration_channel / 1e6) + "MHz \n")
        file.write("Calibration Scheme: " + str(calibration_scheme) + "\n")
        file.write("Noise Parameters: " + str(noise_param) + "\n")
        file.write("Source Direction: " + direction + "\n")
        file.write("Sky Steps: " + str(sky_steps) + "\n")
        file.write("Sky Model: " + str(sky_param) + "\n")
        file.write("Iterations: " + str(iterations) + "\n")
        file.write("Beam Parameters: " + str(beam_param) + "\n")
        file.close()

    endtime = time.time()
    print "Runtime", endtime - starttime
    return


def check_noise_and_sky_parameters(noise_param, sky_param, sky_steps, input_iterations):
    type_sim = ""

    if noise_param[0]:
        sim_iterations = input_iterations
        type_sim += " noisy "
    else:
        sim_iterations = 1
        type_sim += " ideal "

    if sky_param[0] == 'background' or sky_param[0] == 'point_and_background':
        simiterations = input_iterations

    if sky_param[0] == "point" or sky_param[0] == 'point_and_background':
        if sky_param[0] == "point":
            type_sky = "%s Jy point source sky" % str(sky_param[1])
        elif sky_param[0] == 'point_and_background':
            type_sky = "%s Jy point source and background sky" % str(sky_param[1])

    elif sky_param[0] == 'background':
        sky_steps = 1
        type_sky = 'background sky'
    else:
        sys.exit(sky_param[0] + " is an invalid sky model parameter. Please " + \
                 "choose from 'point' or 'background' or 'point_and_background'")

    return sim_iterations, sky_steps, type_sim


def source_flux_and_position_offset_changer(telescope_param,calibration_channel,noise_param, sky_param, beam_param,
                                              calibration_scheme, peakflux_range, offset_range,iterations,save_to_disk):
    """
    """

    print "Simulating the Calibration of Arrays with Redundancy"
    print "Changing Maximum Flux and Position offsets"
    start_time = time.time()

    minimum_position_offset = numpy.log10(offset_range[0])
    maximum_position_offset = numpy.log10(offset_range[1])
    position_step_number = offset_range[2]
    position_offsets = numpy.logspace(minimum_position_offset,maximum_position_offset, position_step_number)

    minimum_peakflux = numpy.log10(peakflux_range[0])
    maximum_peakflux = numpy.log10(peakflux_range[1])
    peakflux_step_number = peakflux_range[2]
    peak_fluxes = numpy.logspace(minimum_peakflux,maximum_peakflux, peakflux_step_number)

    random_seeds = numpy.arange(iterations)

    #generate idealized telescope coordinates
    if telescope_param[0] == 'square' \
    or telescope_param[0] == 'hex' \
    or telescope_param[0] == 'doublehex' \
    or telescope_param[0] == 'doublesquare' \
    or telescope_param[0] == 'linear':
        xyz_positions = xyz_position_creator(telescope_param)
    else:
        xyz_positions = antenna_table_loader(telescope_param[0])

    #generate antenna gains
    frequency_range = numpy.array(calibration_channel)
    gain_table = antenna_gain_creator(xyz_positions, frequency_range)

    #Create an initial baseline tables to identify which parameters we're going to solve for.
    baseline_table = baseline_converter(xyz_positions,gain_table,frequency_range)
    red_baseline_table = redundant_baseline_finder(baseline_table,'ALL', verbose=True)
    amp_matrix, phase_matrix, red_tiles, red_groups = LogcalMatrixPopulator(red_baseline_table,xyz_positions)

    #Knowing what we're solving for we can start setting up tables
    n_measurements = red_baseline_table.shape[0]
    n_tiles = len(red_tiles)
    n_groups= len(red_groups)
    n_peakfluxes = len(peak_fluxes)
    n_offsets = len(position_offsets)

    noisy_amp_solutions = numpy.zeros((n_tiles+n_groups,n_peakfluxes,n_offsets,iterations))
    noisy_phase_solutions = numpy.zeros((n_tiles+n_groups,n_peakfluxes,n_offsets,iterations))
    ideal_amp_solutions = numpy.zeros((n_tiles+n_groups,n_peakfluxes,n_offsets,iterations))
    ideal_phase_solutions = numpy.zeros((n_tiles+n_groups,n_peakfluxes,n_offsets,iterations))

    iteration_counter = 0
    for iteration in range(iterations):
        if numpy.mod(iteration, 100) == 0:
            print "Realization", iteration

        sigma_counter = 0
        for sigma in position_offsets:

            #We want to generate an array which has offset but is still completely redundant!
            array_counter = 0
            while True:
                #Generate positions offsets to add to the antenna positions
                x_offset = numpy.random.normal(0, sigma, gain_table[:, 1].shape[0])
                y_offset = numpy.random.normal(0, sigma, gain_table[:, 2].shape[0])

                offset_positions = xyz_positions.copy()
                offset_positions[:,1] += x_offset
                offset_positions[:,2] += y_offset

                offset_baseline_table = baseline_converter(offset_positions, gain_table, frequency_range,verbose = False)
                off_red_baseline_table = redundant_baseline_finder(offset_baseline_table,'ALL')

                if off_red_baseline_table.shape == red_baseline_table.shape:
                    amp_matrix, phase_matrix, off_red_tiles, off_red_groups = LogcalMatrixPopulator(off_red_baseline_table,
                                                                                                offset_positions)
                    #now we check whether the number of redundant tiles and groups is still the same
                    if len(off_red_tiles) == n_tiles and len(off_red_groups) == n_groups:
                        array_succes = True
                        break
                    elif array_counter > 100:
                        array_succes = False
                        break
                else:
                    array_counter += 1

            peakflux_counter = 0
            for peakflux in peak_fluxes:
                #If we managed to create redundant telescope
                #print array_succes
                if array_succes:
                    sky_model = [sky_param[0], peakflux, sky_param[2],sky_param[3]]
                    obs_visibilities, ideal_visibilities, model_visibilities = \
                        CreateVisibilities(off_red_baseline_table,frequency_range,noise_param,sky_model,beam_param,
                                           random_seeds[iteration_counter])

                    if calibration_scheme == 'lincal':
                        true_solutions = TrueSolutions_Organizer(gain_table,model_visibilities,off_red_baseline_table,
                                                                 off_red_tiles,off_red_groups)
                        calibration_param = ['lincal', true_solutions]
                    elif calibration_scheme == 'logcal' or calibration_scheme == 'full':
                        calibration_param = [calibration_scheme]
                    else:
                        sys.exit("INVALID PARAMETER -calibration_scheme: 'logcal','lincal' or 'full'")

                    #Pass the visibility data and calibration parameters along to the calibrator
                    noisy_amp_results, noisy_phase_results = \
                        Redundant_Calibrator(amp_matrix, phase_matrix, obs_visibilities, off_red_baseline_table,
                                             off_red_tiles, off_red_groups, calibration_param)
                    ideal_amp_results, ideal_phase_results = \
                        Redundant_Calibrator(amp_matrix, phase_matrix, ideal_visibilities, off_red_baseline_table,
                                             off_red_tiles, off_red_groups, calibration_param)

                    noisy_amp_solutions[:,  peakflux_counter, sigma_counter, iteration_counter] = noisy_amp_results
                    noisy_phase_solutions[:,  peakflux_counter, sigma_counter, iteration_counter] = noisy_phase_results
                    ideal_amp_solutions[:,  peakflux_counter, sigma_counter, iteration_counter] = ideal_amp_results
                    ideal_phase_solutions[:,  peakflux_counter, sigma_counter, iteration_counter] = ideal_phase_results
                else:
                    noisy_amp_solutions[:,  peakflux_counter, sigma_counter, iteration_counter] = numpy.nan
                    noisy_phase_solutions[:,  peakflux_counter, sigma_counter, iteration_counter] = numpy.nan
                    ideal_amp_solutions[:,  peakflux_counter, sigma_counter, iteration_counter] = numpy.nan
                    ideal_phase_solutions[:,  peakflux_counter, sigma_counter, iteration_counter] = numpy.nan

                peakflux_counter += 1
            sigma_counter += 1
        iteration_counter += 1

    parameters = numpy.concatenate((red_tiles, red_groups))
    axesdata = [parameters, position_offsets, peak_fluxes, random_seeds]
    axeskeys = ['parameters', 'positions_uncertainty', 'peak_fluxes', 'iteration']
    save_to_hdf5(save_to_disk[1], "SFPO_ideal_amp_solutions", ideal_amp_solutions,
                 axesdata, axeskeys)
    save_to_hdf5(save_to_disk[1], "SFPO_ideal_phase_solutions", ideal_phase_solutions,
                 axesdata, axeskeys)

    save_to_hdf5(save_to_disk[1], "SFPO_noisy_amp_solutions", noisy_amp_solutions,
                 axesdata, axeskeys)
    save_to_hdf5(save_to_disk[1], "SFPO_noisy_phase_solutions", noisy_phase_solutions,
                 axesdata, axeskeys)

    endtime = time.time()
    runtime = endtime - start_time

    file = open(save_to_disk[1] + "SLPO_simulation.log","w")
    file.write("Moving source and changing position offset simulation")
    file.write("Telescope Parameters: " + str(telescope_param) + "\n")
    file.write("Calibration Channel: " + str(frequency_range / 1e6) + "MHz \n")
    file.write("Noise Parameters: " + str(noise_param) + "\n")
    file.write("Sky Model: " + str(sky_param) + "\n")
    file.write("Beam Parameters: " + str(beam_param) + "\n")
    file.write("Calibration scheme: " + str(calibration_scheme) + "\n")
    file.write("Offset Range: " + str(offset_range) + "\n")
    file.write("Peak Flux Range: " + str(peakflux_range) + "\n")
    file.write("Iterations: " + str(iterations) + "\n")
    file.write("Runtime: " + str(runtime) + "\n")
    file.close()
    print "Runtime", runtime
    return


def moving_source_and_position_offset_changer(telescope_param, calibration_channel, noise_param, sky_param, beam_param,
                                              calibration_scheme, source_position_range, offset_range, iterations, save_to_disk):
    """
    """
    print "Simulating the Calibration of Arrays with Redundancy (SCAR)"
    print "Changing source position and position offsets"

    start_time = time.time()

    source_positions = numpy.linspace(source_position_range[0], source_position_range[1], source_position_range[2])
    minimum_position_offset = numpy.log10(offset_range[0])
    maximum_position_offset = numpy.log10(offset_range[1])
    position_step_number = offset_range[2]
    position_offsets = numpy.logspace(minimum_position_offset,maximum_position_offset, position_step_number)

    random_seeds = numpy.arange(iterations)

    #generate idealized telescope coordinates
    if telescope_param[0] == 'square' \
    or telescope_param[0] == 'hex' \
    or telescope_param[0] == 'doublehex' \
    or telescope_param[0] == 'doublesquare' \
    or telescope_param[0] == 'linear':
        xyz_positions = xyz_position_creator(telescope_param)
    else:
        xyz_positions = antenna_table_loader(telescope_param[0])

    #generate antenna gains
    frequency_range = numpy.array(calibration_channel)
    gain_table = antenna_gain_creator(xyz_positions, frequency_range)

    #Create an initial baseline tables to identify which parameters we're going to solve for.
    baseline_table = baseline_converter(xyz_positions,gain_table,frequency_range)
    red_baseline_table = redundant_baseline_finder(baseline_table,'ALL', verbose=True)
    amp_matrix, phase_matrix, red_tiles, red_groups = LogcalMatrixPopulator(red_baseline_table,xyz_positions)

    #Knowing what we're solving for we can start setting up tables
    n_measurements = red_baseline_table.shape[0]
    n_tiles = len(red_tiles)
    n_groups= len(red_groups)
    n_coordinates = len(source_positions)
    n_offsets = len(position_offsets)

    noisy_amp_solutions = numpy.zeros((n_tiles+n_groups,n_coordinates,n_offsets,iterations))
    noisy_phase_solutions = numpy.zeros((n_tiles+n_groups,n_coordinates,n_offsets,iterations))
    ideal_amp_solutions = numpy.zeros((n_tiles+n_groups,n_coordinates,n_offsets,iterations))
    ideal_phase_solutions = numpy.zeros((n_tiles+n_groups,n_coordinates,n_offsets,iterations))

    iteration_counter = 0
    for iteration in range(iterations):
        if numpy.mod(iteration, 100) == 0:
            print "Realization", iteration

        sigma_counter = 0
        for sigma in position_offsets:

            #We want to generate an array which has offset but is still completely redundant!
            array_counter = 0
            while True:
                #Generate positions offsets to add to the antenna positions
                x_offset = numpy.random.normal(0, sigma, gain_table[:, 1].shape[0])
                y_offset = numpy.random.normal(0, sigma, gain_table[:, 2].shape[0])

                offset_positions = xyz_positions.copy()
                offset_positions[:,1] += x_offset
                offset_positions[:,2] += y_offset

                offset_baseline_table = baseline_converter(offset_positions, gain_table, frequency_range,verbose = False)
                off_red_baseline_table = redundant_baseline_finder(offset_baseline_table,'ALL')

                if off_red_baseline_table.shape == red_baseline_table.shape:
                    amp_matrix, phase_matrix, off_red_tiles, off_red_groups = LogcalMatrixPopulator(off_red_baseline_table,
                                                                                                offset_positions)
                    #now we check whether the number of redundant tiles and groups is still the same
                    if len(off_red_tiles) == n_tiles and len(off_red_groups) == n_groups:
                        array_succes = True
                        break
                    elif array_counter > 100:
                        array_succes = False
                        break
                else:
                    array_counter += 1

            location_counter = 0
            for source_location in source_positions:
                #If we managed to create redundant telescope
                #print array_succes
                if array_succes:
                    sky_model = [sky_param[0],sky_param[1],source_location,sky_param[3]]
                    obs_visibilities, ideal_visibilities, model_visibilities = \
                        CreateVisibilities(off_red_baseline_table,frequency_range,noise_param,sky_model,beam_param,
                                           random_seeds[iteration_counter])

                    if calibration_scheme == 'lincal':
                        true_solutions = TrueSolutions_Organizer(gain_table,model_visibilities,off_red_baseline_table,
                                                                 off_red_tiles,off_red_groups)
                        calibration_param = ['lincal', true_solutions]
                    elif calibration_scheme == 'logcal' or calibration_scheme == 'full':
                        calibration_param = [calibration_scheme]
                    else:
                        sys.exit("INVALID PARAMETER -calibration_scheme: 'logcal','lincal' or 'full'")

                    #Pass the visibility data and calibration parameters along to the calibrator
                    noisy_amp_results, noisy_phase_results = \
                        Redundant_Calibrator(amp_matrix, phase_matrix, obs_visibilities, off_red_baseline_table,
                                             off_red_tiles, off_red_groups, calibration_param)
                    ideal_amp_results, ideal_phase_results = \
                        Redundant_Calibrator(amp_matrix, phase_matrix, ideal_visibilities, off_red_baseline_table,
                                             off_red_tiles, off_red_groups, calibration_param)

                    noisy_amp_solutions[:,  location_counter, sigma_counter, iteration_counter] = noisy_amp_results
                    noisy_phase_solutions[:,  location_counter, sigma_counter, iteration_counter] = noisy_phase_results
                    ideal_amp_solutions[:,  location_counter, sigma_counter, iteration_counter] = ideal_amp_results
                    ideal_phase_solutions[:,  location_counter, sigma_counter, iteration_counter] = ideal_phase_results
                else:
                    noisy_amp_solutions[:,  location_counter, sigma_counter, iteration_counter] = numpy.nan
                    noisy_phase_solutions[:,  location_counter, sigma_counter, iteration_counter] = numpy.nan
                    ideal_amp_solutions[:,  location_counter, sigma_counter, iteration_counter] = numpy.nan
                    ideal_phase_solutions[:,  location_counter, sigma_counter, iteration_counter] = numpy.nan

                location_counter += 1
            sigma_counter += 1
        iteration_counter += 1

    parameters = numpy.concatenate((red_tiles, red_groups))
    axesdata = [parameters, position_offsets, source_positions, random_seeds]
    axeskeys = ['parameters', 'positions_uncertainty', 'source_positions', 'iteration']
    save_to_hdf5(save_to_disk[1], "SLPO_ideal_amp_solutions", ideal_amp_solutions,
                 axesdata, axeskeys)
    save_to_hdf5(save_to_disk[1], "SLPO_ideal_phase_solutions", ideal_phase_solutions,
                 axesdata, axeskeys)

    save_to_hdf5(save_to_disk[1], "SLPO_noisy_amp_solutions", noisy_amp_solutions,
                 axesdata, axeskeys)
    save_to_hdf5(save_to_disk[1], "SLPO_noisy_phase_solutions", noisy_phase_solutions,
                 axesdata, axeskeys)

    endtime = time.time()
    runtime = endtime - start_time

    file = open(save_to_disk[1] + "SLPO_simulation.log","w")
    file.write("Moving source and changing position offset simulation")
    file.write("Telescope Parameters: " + str(telescope_param) + "\n")
    file.write("Calibration Channel: " + str(frequency_range / 1e6) + "MHz \n")
    file.write("Noise Parameters: " + str(noise_param) + "\n")
    file.write("Sky Model: " + str(sky_param) + "\n")
    file.write("Beam Parameters: " + str(beam_param) + "\n")
    file.write("Calibration scheme: " + str(calibration_scheme) + "\n")
    file.write("Offset Range: " + str(offset_range) + "\n")
    file.write("Iterations: " + str(iterations) + "\n")
    file.write("Runtime: " + str(runtime) + "\n")
    file.close()
    print "Runtime", runtime
    return
