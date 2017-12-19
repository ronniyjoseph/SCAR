import numpy
import os
import sys
import scipy.constants
from scipy import interpolate
from GeneralTools import unique_value_finder
from GeneralTools import position_finder
from GeneralTools import logcal_solver
from GeneralTools import solution_averager
from GeneralTools import solution_histogram_plotter
from RadioTelescope import antenna_gain_creator
from RadioTelescope import baseline_converter
from RadioTelescope import redundant_baseline_finder
from RadioTelescope import xyz_position_creator
from SkyModel import CreateVisibilities
import time
from matplotlib import pyplot

"""
13 April 2017
-Added Franzen et al. 2016 power law
-Changed lay out: calculate results for a certain track along the sky
 and repeat this a thousand times, then average along the realization 
 direction.
Note: This might use up way more RAM and may be way more inefficient
But maybe more efficient then generating a new sky for each realization
and sky step or storing all sky realization
- Decided NOT to do Vectorization because of heavy memory load
- V5 Added histogramplotter for the solutions and the inputvisibilities
- V6 changing the std into interquantile range to get the bulk 50% width
		Mean instead of mean
- V7 changed back to std and merged ideal and noisy loop, no need for two
		calls with different skies
- Added double hex	
- V8 Multifrequency	
TODO:
xyz_position_creator -> Get the scaling of the double hex correct to match
MWA scale
Finish documentation
Just adding some commit stuff
another 
"""


def MuChSource_Mover(n_channels, telescope_param, calibration_channel, noise_param, direction,
                     sky_steps, sky_param, beam_param, save_to_disk):
    # Track how long it's taking
    starttime = time.time()

    xyz_positions = xyz_position_creator(telescope_param)
    channel_size = noise_param[2]

    # calculate the frequencies of the adjecent channels
    frequency_range = numpy.arange(calibration_channel - n_channels * channel_size,
                                   calibration_channel + (n_channels + 1) * channel_size, channel_size)

    gain_table = antenna_gain_creator(xyz_positions, frequency_range)
    baseline_table = baseline_converter(xyz_positions, gain_table, frequency_range)

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

    noisy_amp_solutions = numpy.zeros((n_tiles + n_groups, sky_steps, iterations))
    noisy_phase_solutions = numpy.zeros((n_tiles + n_groups, sky_steps, iterations))

    ideal_amp_solutions = numpy.zeros((n_tiles + n_groups, sky_steps, iterations))
    ideal_phase_solutions = numpy.zeros((n_tiles + n_groups, sky_steps, iterations))

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
                CreateVisibilities(calibration_baselines, calibration_frequencies, [False], sky_model, beam_param, seed)

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
                    CreateVisibilities(calibration_baselines, calibration_frequencies, noise_param, sky_model,
                                       beam_param,
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
                        calibration_baselines, calibration_frequencies, noise_param, sky_model, beam_param, seed)

                obs_visibilities += point_obs_visibilities
                ideal_visibilities += point_ideal_visibilities
                model_visibilities += point_model_visibilities

            # Create point source data in the absence of background sky data
            elif sky_param[0] == 'point':
                sky_model = ['point', sky_param[1], l, m]
                noise_param[0] = 'SEFD'

                obs_visibilities, ideal_visibilities, \
                model_visibilities = \
                    CreateVisibilities(calibration_baselines, calibration_frequencies, noise_param, sky_model,
                                       beam_param,
                                       seed)

            # Use the model data  to solve for the antenna gains
            ideal_amp_data, ideal_phase_data = MuChRedCalibrator(amp_matrix, phase_matrix, ideal_visibilities)

            ideal_amp_solutions[:, i, j] = ideal_amp_data
            ideal_phase_solutions[:, i, j] = ideal_phase_data

            # Use the noisy data  to solve for the antenna gains
            noisy_amp_data, noisy_phase_data = MuChRedCalibrator(amp_matrix, phase_matrix, obs_visibilities)

            noisy_amp_solutions[:, i, j] = noisy_amp_data
            noisy_phase_solutions[:, i, j] = noisy_phase_data

            # remove the point source
            if sky_param[0] == 'point_and_background':
                obs_visibilities -= point_obs_visibilities
                ideal_visibilities -= point_ideal_visibilities
                model_visibilities -= point_model_visibilities

    noisy_amp_info, noisy_phase_info = solution_averager(
        noisy_amp_solutions, noisy_phase_solutions, red_tiles, red_groups, sky_coords, save_to_disk, direction, [True])
    ideal_amp_info, ideal_phase_info = solution_averager(
        ideal_amp_solutions, ideal_phase_solutions, red_tiles, red_groups, sky_coords, save_to_disk, direction, [False])
    if save_to_disk[0]:
        file = open(save_to_disk[1] + "simulation_parameter.log", "w")
        file.write("Multi-Channel Redundant Calibration Simulation\n")
        file.write("Telescope Parameters: " + str(telescope_param) + "\n")
        file.write("Number Channels: " + str(n_channels) + "\n")
        file.write("Calibration Channel: " + str(calibration_channel / 1e6) + "MHz \n")
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


def MuChMatrixPopulator(baseline_table, xyz_positions):
    n_channels = baseline_table.shape[2] - 1
    middle_index = (1 + n_channels) / 2

    antenna_pairs = baseline_table[:, 0:2, middle_index]
    baseline_groups = baseline_table[:, 7, middle_index]
    # so first we sort out the unique antennas
    # and the unique redudant groups, this will allows us to populate the matrix adequately
    red_tiles = unique_value_finder(antenna_pairs, 'values')
    # it's not really finding unique antennas, it just finds unique values
    red_groups = unique_value_finder(baseline_groups, 'values')

    n_measurements = baseline_table.shape[0]
    n_tiles = len(red_tiles)
    n_groups = len(red_groups)
    print "There are", n_tiles, "redundant tiles"

    # create am empty matrix (#measurements)x(#tiles + #redundant groups)
    single_amp_matrix = numpy.zeros((n_measurements, n_tiles + n_groups))
    single_phase_matrix = numpy.zeros((n_measurements, n_tiles + n_groups))
    for i in range(n_measurements):
        index1 = numpy.where(red_tiles == antenna_pairs[i, 0])
        index2 = numpy.where(red_tiles == antenna_pairs[i, 1])
        index_group = numpy.where(red_groups == baseline_groups[i])

        single_amp_matrix[i, index1[0]] = 1
        single_amp_matrix[i, index2[0]] = 1
        single_amp_matrix[i, n_tiles + index_group[0]] = 1

        single_phase_matrix[i, index1[0]] = -1
        single_phase_matrix[i, index2[0]] = 1
        single_phase_matrix[i, n_tiles + index_group[0]] = 1
    print ""
    print "Creating the equation matrix"
    # Create the multi frequency matrix
    amp_matrix = numpy.zeros((n_channels * n_measurements, n_tiles \
                              + n_channels * n_groups))
    phase_matrix = numpy.zeros((n_channels * n_measurements, n_tiles \
                                + n_channels * n_groups))

    gain_amp_mapper = single_amp_matrix[:, 0:n_tiles]
    vis_amp_mapper = single_amp_matrix[:, n_tiles:]
    gain_phase_mapper = single_phase_matrix[:, 0:n_tiles]
    vis_phase_mapper = single_phase_matrix[:, n_tiles:]

    counter = -n_channels
    MuCh_red_groups = numpy.zeros(n_groups * n_channels)
    for i in range(n_channels):
        amp_matrix[i * (n_measurements):(i + 1) * (n_measurements),
        0:n_tiles] = gain_amp_mapper
        phase_matrix[i * (n_measurements):(i + 1) * (n_measurements),
        0:n_tiles] = gain_phase_mapper

        amp_matrix[i * (n_measurements):(i + 1) * (n_measurements),
        n_tiles + i * n_groups:n_tiles + (1 + i) * n_groups] = \
            vis_amp_mapper
        phase_matrix[i * (n_measurements):(i + 1) * (n_measurements),
        n_tiles + i * n_groups:n_tiles + (1 + i) * n_groups] = \
            vis_phase_mapper
        MuCh_red_groups[i * n_groups:(i + 1) * n_groups] = counter * red_groups

        counter += 1
        if counter == 0:
            counter += 1

    # select the xy-positions for the red_tiles
    red_x_positions, red_y_positions = position_finder(red_tiles, xyz_positions)

    # add this to the amplitude matrix
    amp_constraints = numpy.zeros((n_tiles + n_channels * n_groups))
    amp_constraints[0] = 1.
    amp_matrix = numpy.vstack((amp_matrix, amp_constraints))
    # add these constraints to the phase matrix
    phase_constraints = numpy.zeros((3, n_tiles + n_channels * n_groups))
    phase_constraints[0, 0] = 1
    phase_constraints[1, 0:n_tiles] = red_x_positions
    phase_constraints[2, 0:n_tiles] = red_y_positions
    phase_matrix = numpy.vstack((phase_matrix, phase_constraints))

    # check whether the matrix is ill conditioned
    phase_dagger = numpy.dot(phase_matrix.transpose(), phase_matrix)
    amp_dagger = numpy.dot(amp_matrix.transpose(), amp_matrix)
    if numpy.linalg.det(numpy.dot(numpy.linalg.pinv(amp_dagger), amp_dagger)) == 0:
        print "WARNING: the amplitude solver matrix is singular"

    if numpy.linalg.det(numpy.dot(numpy.linalg.pinv(phase_dagger), phase_dagger)) == 0:
        print "WARNING: the phase solver matrix is singular"

    phase_pinv = numpy.dot(numpy.linalg.pinv(phase_dagger), phase_matrix.transpose())
    amp_pinv = numpy.dot(numpy.linalg.pinv(amp_dagger), amp_matrix.transpose())

    return amp_pinv, phase_pinv, red_tiles, MuCh_red_groups


def MuChRedCalibrator(amp_matrix, phase_matrix, multichannel_visibilities):
    ####################Redundant Calibration###########################
    # Reshape the data before inputting it into the standard logcal solver

    n_measurements = multichannel_visibilities.shape[0]
    n_channels = multichannel_visibilities.shape[1]

    obs_visibilities = numpy.zeros((n_measurements * n_channels), dtype=complex)
    for i in range(n_channels):
        obs_visibilities[i * n_measurements:(i + 1) * n_measurements] = \
            multichannel_visibilities[:, i]

    # feed observations into a general gain solver function
    amp_solutions, phase_solutions = \
        logcal_solver(amp_matrix, phase_matrix, obs_visibilities)

    return amp_solutions, phase_solutions


def MuChFatSource_Mover(n_channels, telescope_param, calibration_channel, noise_param, direction,
                        sky_steps, sky_param, beam_param, save_to_disk, histo_movie):
    # Track how long it's taking
    starttime = time.time()

    xyz_positions = xyz_position_creator(telescope_param)
    channel_size = noise_param[2]

    # calculate the frequencies of the adjecent channels
    # frequency_range = numpy.arange(calibration_channel-n_channels*channel_size,
    # calibration_channel+(n_channels+1)*channel_size,channel_size)

    # Changing the frequency range -> creating two fat channels
    frequency_range = numpy.array([
        calibration_channel - n_channels * channel_size,
        calibration_channel,
        calibration_channel + n_channels * channel_size])

    gain_table = antenna_gain_creator(xyz_positions, frequency_range)
    baseline_table = baseline_converter(xyz_positions, gain_table, frequency_range)
    # change the noise bandwith for those 2 fat channels
    # noise_param[2] = n_channels*channel_size

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

    noisy_amp_solutions = numpy.zeros((n_tiles + n_groups, sky_steps, iterations))
    noisy_phase_solutions = numpy.zeros((n_tiles + n_groups, sky_steps, iterations))

    ideal_amp_solutions = numpy.zeros((n_tiles + n_groups, sky_steps, iterations))
    ideal_phase_solutions = numpy.zeros((n_tiles + n_groups, sky_steps, iterations))

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
                CreateVisibilities(calibration_baselines, calibration_frequencies, [False], sky_model, beam_param, seed)

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
                    CreateVisibilities(calibration_baselines, calibration_frequencies, noise_param, sky_model,
                                       beam_param,
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
                        calibration_baselines, calibration_frequencies, noise_param, sky_model, beam_param, seed)

                obs_visibilities += point_obs_visibilities
                ideal_visibilities += point_ideal_visibilities
                model_visibilities += point_model_visibilities

            # Create point source data in the absence of background sky data
            elif sky_param[0] == 'point':
                sky_model = ['point', sky_param[1], l, m]
                noise_param[0] = 'SEFD'

                obs_visibilities, ideal_visibilities, \
                model_visibilities = \
                    CreateVisibilities(calibration_baselines, calibration_frequencies, noise_param, sky_model,
                                       beam_param,
                                       seed)

            # Use the model data  to solve for the antenna gains
            ideal_amp_data, ideal_phase_data = MuChRedCalibrator(amp_matrix, phase_matrix, ideal_visibilities)

            ideal_amp_solutions[:, i, j] = ideal_amp_data
            ideal_phase_solutions[:, i, j] = ideal_phase_data

            # Use the noisy data  to solve for the antenna gains
            noisy_amp_data, noisy_phase_data = MuChRedCalibrator(amp_matrix, phase_matrix, obs_visibilities)

            noisy_amp_solutions[:, i, j] = noisy_amp_data
            noisy_phase_solutions[:, i, j] = noisy_phase_data

            # remove the point source
            if sky_param[0] == 'point_and_background':
                obs_visibilities -= point_obs_visibilities
                ideal_visibilities -= point_ideal_visibilities
                model_visibilities -= point_model_visibilities

    noisy_amp_info, noisy_phase_info = solution_averager(
        noisy_amp_solutions, noisy_phase_solutions, red_tiles, red_groups, sky_coords, save_to_disk, direction, [True])
    ideal_amp_info, ideal_phase_info = solution_averager(
        ideal_amp_solutions, ideal_phase_solutions, red_tiles, red_groups, sky_coords, save_to_disk, direction, [False])

    if histo_movie[0] == "solutions":
        solution_histogram_plotter(noisy_amp_solutions, noisy_phase_solutions,
                                   noisy_amp_info, noisy_phase_info, histo_movie[1])

    if save_to_disk[0]:
        file = open(save_to_disk[1] + "simulation_parameter.log", "w")
        file.write("Multi-Channel Redundant Calibration Simulation\n")
        file.write("Telescope Parameters: " + str(telescope_param) + "\n")
        file.write("Number Channels: " + str(n_channels) + "\n")
        file.write("Calibration Channel: " + str(calibration_channel / 1e6) + "MHz \n")
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
