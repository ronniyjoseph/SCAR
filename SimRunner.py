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
from GeneralTools import visibility_histogram_plotter
from GeneralTools import solution_histogram_plotter
from GeneralTools import TrueSolutions_Organizer
from SkyModel import CreateVisibilities
from RedundantCalibration import Redundant_Calibrator
from RedundantCalibration import LogcalMatrixPopulator

"""Simulate Calibration with Array Redundancy"""


def Moving_Source(telescope_param, offset, calibration_channel, noise_param, direction,
                  sky_steps, sky_param, beam_param, calibration_scheme, save_to_disk, hist_movie):
    starttime = time.time()

    if telescope_param[0] == 'square' \
            or telescope_param[0] == 'hex' \
            or telescope_param[0] == 'doublehex' \
            or telescope_param[0] == 'doublesquare' \
            or telescope_param[0] == 'linear':
        xyz_positions = xyz_position_creator(telescope_param)
    else:
        xyz_positions = antenna_table_loader(telescope_param[0])

    if offset[0] == True:
        if offset[2] == 'x':
            print "offsetting tile", offset[1], "by", offset[3], "meters"
            xyz_positions[offset[1], 1] += offset[3]
        elif offset[2] == 'y':
            xyz_positions[offset[1], 2] += offset[3]

    frequency_range = numpy.array([calibration_channel])
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

    print ""

    # Find the redundant tiles
    red_baseline_table = redundant_baseline_finder(baseline_table, 'ALL')
    # Calculate the solving matrices (only needs to be once)
    amp_matrix, phase_matrix, red_tiles, red_groups = LogcalMatrixPopulator(
        red_baseline_table, xyz_positions)

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
    n_tiles = len(red_tiles)
    n_groups = len(red_groups)

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
    print "Simulating redundant calibration with a" + type_sim + type_sky

    for j in range(iterations):
        if numpy.mod(j, 100) == 0:
            print "Realization", j
        # seed = numpy.random.randint(1000)
        seed = random_seeds[j]
        if sky_param[0] == "background" or sky_param[0] == 'point_and_background':
            # Create the visibilities for the static background sky
            sky_model = ['background']
            obs_visibilities, ideal_visibilities, model_visibilities = \
                CreateVisibilities(red_baseline_table, frequency_range,
                                   [False], sky_model, beam_param, seed)

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
                point_obs_visibilities, point_ideal_visibilities, point_model_visibilities = \
                    CreateVisibilities(red_baseline_table, frequency_range
                                       , noise_param, sky_model, beam_param, seed)

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
                    CreateVisibilities(red_baseline_table, frequency_range,
                                       noise_param, sky_model, beam_param, seed)

                obs_visibilities += point_obs_visibilities
                ideal_visibilities += point_ideal_visibilities
                model_visibilities += point_model_visibilities

            # Create point source data in the absence of background sky data
            elif sky_param[0] == 'point':
                sky_model = ['point', sky_param[1], l, m]
                noise_param[0] = 'source'
                obs_visibilities, ideal_visibilities, model_visibilities = \
                    CreateVisibilities(red_baseline_table, frequency_range
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

    noisy_amp_info, noisy_phase_info = solution_averager(
        noisy_amp_solutions, noisy_phase_solutions, red_tiles, red_groups,
        sky_coords, save_to_disk, direction, [True])
    ideal_amp_info, ideal_phase_info = solution_averager(
        ideal_amp_solutions, ideal_phase_solutions, red_tiles, red_groups,
        sky_coords, save_to_disk, direction, [False])
    if save_to_disk[0]:
        file = open(save_to_disk[1] + "simulation_parameter.log", "w")
        file.write("Standard Redundant Calibration Simulation" + "\n")
        file.write("Telescope Parameters: " + str(telescope_param) + "\n")
        file.write("Telescope Offsets: " + str(offset) + "\n")
        file.write("Calibration Channel: " + str(calibration_channel / 1e6) + "MHz \n")
        file.write("Noise Parameters: " + str(noise_param) + "\n")
        file.write("Source Direction: " + direction + "\n")
        file.write("Sky Steps: " + str(sky_steps) + "\n")
        file.write("Sky Model: " + str(sky_param) + "\n")
        file.write("Iterations: " + str(iterations) + "\n")
        file.write("Beam Parameters: " + str(beam_param) + "\n")
        file.close()

    if hist_movie[0]:
        if noise_param[0]:
            # ~ solution_histogram_plotter(noisy_amp_solutions,
            # ~ noisy_phase_solutions, noisy_amp_info, noisy_phase_info,
            # ~ hist_movie[1])

            visibility_histogram_plotter(amp_obs, phase_obs, amp_mod,
                                         phase_mod, sky_coords, noisy_amp_info, noisy_phase_info,
                                         hist_movie[1])

            solution_histogram_plotter(noisy_amp_solutions,
                                       noisy_phase_solutions, noisy_amp_info, noisy_phase_info,
                                       hist_movie[1])

    endtime = time.time()
    print "Runtime", endtime - starttime
    return


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


def max_source_and_position_offset_changer(telescope_param, calibration_channel, noise_param,
    sky_param, beam_param, calibration_scheme, save_to_disk, hist_movie):
    starttime = time.time()

    if telescope_param[0] == 'square' \
            or telescope_param[0] == 'hex' \
            or telescope_param[0] == 'doublehex' \
            or telescope_param[0] == 'doublesquare' \
            or telescope_param[0] == 'linear':
        xyz_positions = xyz_position_creator(telescope_param)
    else:
        xyz_positions = antenna_table_loader(telescope_param[0])

    frequency_range = numpy.array([calibration_channel])
    gain_table = antenna_gain_creator(xyz_positions, frequency_range)

    for sigma in range(offset_range):
        for S_peak in range(peak_range) :
            for iteration in range(n_iterations):

                x_offset = numpy.random.normal(0,sigma,gain_table[:,1].shape)
                y_offset = numpy.random.normal(0, sigma, gain_table[:, 2].shape)
                gain_table[:,1]+= x_offset
                gain_table[:,2]+= y_offset

                baseline_table = baseline_converter(xyz_positions, gain_table,
                                                    frequency_range)
                red_baseline_table = redundant_baseline_finder(baseline_table, 'ALL')
                # Calculate the solving matrices (only needs to be once)
                amp_matrix, phase_matrix, red_tiles, red_groups = LogcalMatrixPopulator(
                    red_baseline_table, xyz_positions)