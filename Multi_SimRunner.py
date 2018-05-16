import numpy
import os
import sys
import time
from functools import partial
import multiprocessing
from RadioTelescope import antenna_gain_creator
from RadioTelescope import baseline_converter
from RadioTelescope import redundant_baseline_finder
from RadioTelescope import xyz_position_creator
from RadioTelescope import antenna_table_loader
from GeneralTools import TrueSolutions_Organizer
from GeneralTools import save_to_hdf5
from GeneralTools import solution_mapper
from SkyModel import analytic_visibilities
from SkyModel import numerical_visibilities
from RedundantCalibration import Redundant_Calibrator
from RedundantCalibration import LogcalMatrixPopulator


def source_flux_and_position_offset_changer_parallel(telescope_param, calibration_channel, noise_param, sky_param,
                                                     beam_param,
                                                     calibration_scheme, peakflux_range, offset_range, n_iterations,
                                                     save_to_disk,
                                                     processes):
    """
    """

    print "Simulating the Calibration of Arrays with Redundancy"
    print "Changing Maximum Flux and Position offsets"
    start_time = time.time()

    if not os.path.exists(save_to_disk[1]):
        print ""
        print "!!!Warning: Creating output folder at output destination!"
        os.makedirs(save_to_disk[1])
        output_types = ["ideal_amp", "ideal_phase", "noisy_amp", "noisy_phase"]
        for output in output_types:
            os.makedirs(save_to_disk[1] + "threaded_" + output + "/")

    minimum_position_offset = numpy.log10(offset_range[0])
    maximum_position_offset = numpy.log10(offset_range[1])
    position_step_number = offset_range[2]
    position_offsets = numpy.logspace(minimum_position_offset, maximum_position_offset, position_step_number)

    minimum_peakflux = numpy.log10(peakflux_range[0])
    maximum_peakflux = numpy.log10(peakflux_range[1])
    peakflux_step_number = peakflux_range[2]
    peak_fluxes = numpy.logspace(minimum_peakflux, maximum_peakflux, peakflux_step_number)

    iterations = numpy.arange(n_iterations)

    # generate idealized telescope coordinates
    if telescope_param[0] == 'square' \
            or telescope_param[0] == 'hex' \
            or telescope_param[0] == 'doublehex' \
            or telescope_param[0] == 'doublesquare' \
            or telescope_param[0] == 'linear':
        xyz_positions = xyz_position_creator(telescope_param)
    else:
        xyz_positions = antenna_table_loader(telescope_param[0])

    # generate antenna gains
    frequency_range = numpy.array(calibration_channel)
    gain_table = antenna_gain_creator(xyz_positions, frequency_range)

    # Create an initial baseline tables to identify which parameters we're going to solve for.
    baseline_table = baseline_converter(xyz_positions, gain_table, frequency_range)
    red_baseline_table = redundant_baseline_finder(baseline_table, 'ALL', verbose=True)
    amp_matrix, phase_matrix, red_tiles, red_groups = LogcalMatrixPopulator(red_baseline_table, xyz_positions)

    pool = multiprocessing.Pool(processes=processes)
    iterator = partial(single_iteration_source_flux_position_offset,
                       xyz_positions, gain_table, frequency_range, peak_fluxes, position_offsets, calibration_scheme,
                       sky_param, noise_param, beam_param, save_to_disk, red_tiles, red_groups, n_iterations)

    pool.map(iterator, iterations)
    end_time = time.time()

    runtime = end_time - start_time
    print "Runtime", runtime
    file = open(save_to_disk[1] + "SFPO_simulation_parameters.log", "w")
    file.write("Changing Source Flux and Position Offset simulation\n")
    file.write("Re-Realising Every Array\n")
    file.write("Telescope Parameters: " + str(telescope_param) + "\n")
    file.write("Calibration Channel: " + str(frequency_range / 1e6) + "MHz \n")
    file.write("Noise Parameters: " + str(noise_param) + "\n")
    file.write("Sky Model: " + str(sky_param) + "\n")
    file.write("Beam Parameters: " + str(beam_param) + "\n")
    file.write("Calibration scheme: " + str(calibration_scheme) + "\n")
    file.write("Offset Range: " + str(offset_range) + "\n")
    file.write("Peak Flux Range: " + str(peakflux_range) + "\n")
    file.write("Iterations: " + str(n_iterations) + "\n")
    file.write("Runtime: " + str(runtime) + "\n")
    file.close()
    return


def single_iteration_source_flux_position_offset(xyz_positions, gain_table, frequency_range, peak_fluxes,
                                                 position_offsets, calibration_scheme, sky_param,
                                                 noise_param,
                                                 beam_param, save_to_disk, red_tiles, red_groups,
                                                 n_iterations, iteration):

    parameters = numpy.concatenate((red_tiles, red_groups))

    noisy_amp_solutions = numpy.zeros((len(parameters), len(position_offsets), len(peak_fluxes)))
    noisy_phase_solutions = numpy.zeros((len(parameters), len(position_offsets), len(peak_fluxes)))
    ideal_amp_solutions = numpy.zeros((len(parameters), len(position_offsets), len(peak_fluxes)))
    ideal_phase_solutions = numpy.zeros((len(parameters), len(position_offsets), len(peak_fluxes)))

    for offset_index in range(len(position_offsets)):
        offset_positions = xyz_positions.copy()
        xy_offsets = numpy.random.normal(0, 1, xyz_positions[:, 1:3].shape)

        offset_positions[:, 1:3] += xy_offsets * position_offsets[offset_index]

        offset_baseline_table = baseline_converter(offset_positions, gain_table, frequency_range, verbose=False)
        off_red_baseline_table = redundant_baseline_finder(offset_baseline_table, 'ALL')

        if off_red_baseline_table.shape[0] == 0:
            empty_results = numpy.zeros(noisy_amp_solutions[:, offset_index, :].shape)
            empty_results[:] = numpy.nan

            noisy_amp_solutions[:, offset_index, :] = empty_results
            noisy_phase_solutions[:, offset_index, :] = empty_results
            ideal_amp_solutions[:, offset_index, :] = empty_results
            ideal_phase_solutions[:, offset_index, :] = empty_results

        else:
            if sky_param[0] == "point_and_background":
                background_model = ['background']
                obs_background, ideal_background, model_background = \
                    numerical_visibilities(off_red_baseline_table, frequency_range, noise_param, background_model,
                                           beam_param, iteration)

            for flux_index in range(len(peak_fluxes)):
                if sky_param[0] == "point":
                    sky_model = [sky_param[0], peak_fluxes[flux_index], sky_param[2], sky_param[3]]
                    obs_visibilities, ideal_visibilities, model_visibilities = \
                        analytic_visibilities(off_red_baseline_table, frequency_range, noise_param, sky_model,
                                              beam_param, iteration)
                elif sky_param[0] == "point_and_background":
                    sky_model = ["point", peak_fluxes[flux_index], sky_param[2], sky_param[3]]
                    obs_point_source, ideal_point_source, model_point_source = \
                        analytic_visibilities(off_red_baseline_table, frequency_range, noise_param, sky_model,
                                              beam_param, iteration)

                    obs_visibilities = obs_background + obs_point_source
                    ideal_visibilities = ideal_background + ideal_point_source
                    model_visibilities = model_background + model_point_source

                amp_matrix, phase_matrix, off_red_tiles, off_red_groups = LogcalMatrixPopulator(off_red_baseline_table,
                                                                                                offset_positions)
                offset_parameters = numpy.concatenate((off_red_tiles, off_red_groups))

                if calibration_scheme == 'lincal':
                    true_solutions = TrueSolutions_Organizer(gain_table, model_visibilities, off_red_baseline_table,
                                                             off_red_tiles, off_red_groups)
                    calibration_param = ['lincal', true_solutions]
                elif calibration_scheme == 'logcal' or calibration_scheme == 'full':
                    calibration_param = [calibration_scheme]
                else:
                    sys.exit("INVALID PARAMETER -calibration_scheme: 'logcal','lincal' or 'full'")

                # Pass the visibility data and calibration parameters along to the calibrator
                noisy_amp_results, noisy_phase_results = \
                    Redundant_Calibrator(amp_matrix, phase_matrix, obs_visibilities, off_red_baseline_table,
                                         off_red_tiles, off_red_groups, calibration_param)
                ideal_amp_results, ideal_phase_results = \
                    Redundant_Calibrator(amp_matrix, phase_matrix, ideal_visibilities, off_red_baseline_table,
                                         off_red_tiles, off_red_groups, calibration_param)
                # map the solutions from the offset array to the ideally redundant array
                noisy_amp_solutions[:, offset_index, flux_index] = solution_mapper(parameters, offset_parameters,
                                                                                   noisy_amp_results)
                noisy_phase_solutions[:, offset_index, flux_index] = solution_mapper(parameters, offset_parameters,
                                                                                     noisy_phase_results)
                ideal_amp_solutions[:, offset_index, flux_index] = solution_mapper(parameters, offset_parameters,
                                                                                   ideal_amp_results)
                ideal_phase_solutions[:, offset_index, flux_index] = solution_mapper(parameters, offset_parameters,
                                                                                     ideal_phase_results)

                # Subtract the point source
                if sky_param[0] == "point_and_background":
                    obs_visibilities -= obs_point_source
                    ideal_visibilities -= ideal_point_source
                    model_visibilities -= model_point_source

    axesdata = [parameters, position_offsets, peak_fluxes]
    axeskeys = ['parameters', 'positions_uncertainty', 'peak_fluxes']
    output_types = ["ideal_amp", "ideal_phase", "noisy_amp", "noisy_phase"]

    prefix = str(0) * (len(str(n_iterations)) - len(str(iteration))) + str(iteration)

    save_to_hdf5(save_to_disk[1] + "threaded_" + output_types[0] + "/", prefix + "_SFPO_ideal_amp_solutions",
                 ideal_amp_solutions, axesdata, axeskeys)
    save_to_hdf5(save_to_disk[1] + "threaded_" + output_types[1] + "/", prefix + "_SFPO_ideal_phase_solutions",
                 ideal_phase_solutions, axesdata, axeskeys)
    save_to_hdf5(save_to_disk[1] + "threaded_" + output_types[2] + "/", prefix + "_SFPO_noisy_amp_solutions",
                 noisy_amp_solutions, axesdata, axeskeys)
    save_to_hdf5(save_to_disk[1] + "threaded_" + output_types[3] + "/", prefix + "_SFPO_noisy_phase_solutions",
                 noisy_phase_solutions, axesdata, axeskeys)

    return


def source_location_and_position_offset_changer_parallel(telescope_param, calibration_channel, noise_param, sky_param,
                                                         beam_param,
                                                         calibration_scheme, source_position_range, offset_range,
                                                         n_iterations,
                                                         save_to_disk,
                                                         processes):
    """
    """
    print "Simulating the Calibration of Arrays with Redundancy"
    print "Changing Source Location and Position offsets"
    start_time = time.time()

    if not os.path.exists(save_to_disk[1]):
        print ""
        print "!!!Warning: Creating output folder at output destination!"
        os.makedirs(save_to_disk[1])
        output_types = ["ideal_amp", "ideal_phase", "noisy_amp", "noisy_phase"]
        for output in output_types:
            os.makedirs(save_to_disk[1] + "threaded_" + output + "/")

    minimum_position_offset = numpy.log10(offset_range[0])
    maximum_position_offset = numpy.log10(offset_range[1])
    position_step_number = offset_range[2]
    position_offsets = numpy.logspace(minimum_position_offset, maximum_position_offset, position_step_number)

    source_locations = numpy.linspace(source_position_range[0], source_position_range[1], source_position_range[2])

    iterations = numpy.arange(n_iterations)

    # generate idealized telescope coordinates
    if telescope_param[0] == 'square' \
            or telescope_param[0] == 'hex' \
            or telescope_param[0] == 'doublehex' \
            or telescope_param[0] == 'doublesquare' \
            or telescope_param[0] == 'linear':
        xyz_positions = xyz_position_creator(telescope_param)
    else:
        xyz_positions = antenna_table_loader(telescope_param[0])

    # generate antenna gains
    frequency_range = numpy.array(calibration_channel)
    gain_table = antenna_gain_creator(xyz_positions, frequency_range)

    # Create an initial baseline tables to identify which parameters we're going to solve for.
    baseline_table = baseline_converter(xyz_positions, gain_table, frequency_range)
    red_baseline_table = redundant_baseline_finder(baseline_table, 'ALL', verbose=True)
    amp_matrix, phase_matrix, red_tiles, red_groups = LogcalMatrixPopulator(red_baseline_table, xyz_positions)

    pool = multiprocessing.Pool(processes=processes)
    iterator = partial(single_iteration_source_location_position_offset,
                       xyz_positions, gain_table, frequency_range, source_locations, position_offsets,
                       calibration_scheme, sky_param, noise_param, beam_param, save_to_disk, red_tiles, red_groups,
                       n_iterations)

    pool.map(iterator, iterations)
    end_time = time.time()

    runtime = end_time - start_time
    print "Runtime", runtime

    file = open(save_to_disk[1] + "SLPO_simulation_parameters.log", "w")
    file.write("Changing Source Location and Position Offset simulation\n")
    file.write("Re-Realising Every Array\n")
    file.write("Telescope Parameters: " + str(telescope_param) + "\n")
    file.write("Calibration Channel: " + str(frequency_range / 1e6) + "MHz \n")
    file.write("Noise Parameters: " + str(noise_param) + "\n")
    file.write("Sky Model: " + str(sky_param) + "\n")
    file.write("Beam Parameters: " + str(beam_param) + "\n")
    file.write("Calibration scheme: " + str(calibration_scheme) + "\n")
    file.write("Offset Range: " + str(offset_range) + "\n")
    file.write("Source location parameters: " + str(source_position_range) + "\n")
    file.write("Iterations: " + str(n_iterations) + "\n")
    file.write("Runtime: " + str(runtime) + "\n")
    file.close()
    return


def single_iteration_source_location_position_offset(xyz_positions, gain_table, frequency_range,
                                                     source_locations, position_offsets, calibration_scheme,
                                                     sky_param, noise_param,
                                                     beam_param, save_to_disk, red_tiles, red_groups, n_iterations,
                                                     iteration):
    parameters = numpy.concatenate((red_tiles, red_groups))

    noisy_amp_solutions = numpy.zeros((len(parameters), len(position_offsets), len(source_locations)))
    noisy_phase_solutions = numpy.zeros((len(parameters), len(position_offsets), len(source_locations)))
    ideal_amp_solutions = numpy.zeros((len(parameters), len(position_offsets), len(source_locations)))
    ideal_phase_solutions = numpy.zeros((len(parameters), len(position_offsets), len(source_locations)))

    for offset_index in range(len(position_offsets)):
        offset_positions = xyz_positions.copy()
        xy_offsets = numpy.random.normal(0, 1, xyz_positions[:, 1:3].shape)

        offset_positions[:, 1:3] += xy_offsets * position_offsets[offset_index]
        offset_baseline_table = baseline_converter(offset_positions, gain_table, frequency_range, verbose=False)
        off_red_baseline_table = redundant_baseline_finder(offset_baseline_table, 'ALL')

        if off_red_baseline_table.shape[0] == 0:
            empty_results = numpy.zeros(noisy_amp_solutions[:, offset_index, :].shape)
            empty_results[:] = numpy.nan

            noisy_amp_solutions[:, offset_index, :] = empty_results
            noisy_phase_solutions[:, offset_index, :] = empty_results
            ideal_amp_solutions[:, offset_index, :] = empty_results
            ideal_phase_solutions[:, offset_index, :] = empty_results

        else:
            if sky_param[0] == "point_and_background":
                background_model = ['background']
                obs_background, ideal_background, model_background = \
                    numerical_visibilities(off_red_baseline_table, frequency_range, noise_param, background_model,
                                           beam_param, iteration)

            for location_index in range(len(source_locations)):
                if sky_param[0] == "point":
                    sky_model = ["point", sky_param[1], source_locations[location_index], sky_param[3]]
                    obs_visibilities, ideal_visibilities, model_visibilities = \
                        analytic_visibilities(off_red_baseline_table, frequency_range, noise_param, sky_model,
                                              beam_param, iteration)
                elif sky_param[0] == "point_and_background":
                    sky_model = ["point", sky_param[1], source_locations[location_index], sky_param[3]]
                    obs_point_source, ideal_point_source, model_point_source = \
                        analytic_visibilities(off_red_baseline_table, frequency_range, noise_param, sky_model,
                                              beam_param, iteration)

                    obs_visibilities = obs_background + obs_point_source
                    ideal_visibilities = ideal_background + ideal_point_source
                    model_visibilities = model_background + model_point_source

                amp_matrix, phase_matrix, off_red_tiles, off_red_groups = LogcalMatrixPopulator(off_red_baseline_table,
                                                                                                offset_positions)
                offset_parameters = numpy.concatenate((off_red_tiles, off_red_groups))

                if calibration_scheme == 'lincal':
                    true_solutions = TrueSolutions_Organizer(gain_table, model_visibilities, off_red_baseline_table,
                                                             off_red_tiles, off_red_groups)
                    calibration_param = ['lincal', true_solutions]
                elif calibration_scheme == 'logcal' or calibration_scheme == 'full':
                    calibration_param = [calibration_scheme]
                else:
                    sys.exit("INVALID PARAMETER -calibration_scheme: 'logcal','lincal' or 'full'")

                # Pass the visibility data and calibration parameters along to the calibrator
                noisy_amp_results, noisy_phase_results = \
                    Redundant_Calibrator(amp_matrix, phase_matrix, obs_visibilities, off_red_baseline_table,
                                         off_red_tiles, off_red_groups, calibration_param)
                ideal_amp_results, ideal_phase_results = \
                    Redundant_Calibrator(amp_matrix, phase_matrix, ideal_visibilities, off_red_baseline_table,
                                         off_red_tiles, off_red_groups, calibration_param)

                # map the solutions from the offset array to the ideally redundant array
                noisy_amp_solutions[:, offset_index, location_index] = solution_mapper(parameters, offset_parameters,
                                                                                       noisy_amp_results)
                noisy_phase_solutions[:, offset_index, location_index] = solution_mapper(parameters, offset_parameters,
                                                                                         noisy_phase_results)
                ideal_amp_solutions[:, offset_index, location_index] = solution_mapper(parameters, offset_parameters,
                                                                                       ideal_amp_results)
                ideal_phase_solutions[:, offset_index, location_index] = solution_mapper(parameters, offset_parameters,
                                                                                         ideal_phase_results)

                # Subtract the point source
                if sky_param[0] == "point_and_background":
                    obs_visibilities -= obs_point_source
                    ideal_visibilities -= ideal_point_source
                    model_visibilities -= model_point_source

    axesdata = [parameters, position_offsets, source_locations]
    axeskeys = ['parameters', 'positions_uncertainty', 'source_locations']
    output_types = ["ideal_amp", "ideal_phase", "noisy_amp", "noisy_phase"]

    prefix = str(0) * (len(str(n_iterations)) - len(str(iteration))) + str(iteration)

    save_to_hdf5(save_to_disk[1] + "threaded_" + output_types[0] + "/", prefix + "_SLPO_ideal_amp_solutions",
                 ideal_amp_solutions, axesdata, axeskeys)
    save_to_hdf5(save_to_disk[1] + "threaded_" + output_types[1] + "/", prefix + "_SLPO_ideal_phase_solutions",
                 ideal_phase_solutions, axesdata, axeskeys)
    save_to_hdf5(save_to_disk[1] + "threaded_" + output_types[2] + "/", prefix + "_SLPO_noisy_amp_solutions",
                 noisy_amp_solutions, axesdata, axeskeys)
    save_to_hdf5(save_to_disk[1] + "threaded_" + output_types[3] + "/", prefix + "_SLPO_noisy_phase_solutions",
                 noisy_phase_solutions, axesdata, axeskeys)

    return


def source_location_changer_MP(telescope_param, offset_param, calibration_channel, noise_param, direction,
                  sky_steps, source_position_range, iterations, sky_param, beam_param, calibration_scheme, save_to_disk,
                               processes):

    if not os.path.exists(save_to_disk[1]):
        print ""
        print "!!!Warning: Creating output folder at output destination!"
        os.makedirs(save_to_disk[1])
        output_types = ["ideal_amp", "ideal_phase", "noisy_amp", "noisy_phase"]
        for output in output_types:
            os.makedirs(save_to_disk[1] + "threaded_" + output + "/")

    starttime = time.time()

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

    source_locations = numpy.linspace(source_position_range[0], source_position_range[1], source_position_range[2])

    pool = multiprocessing.Pool(processes=processes)
    iterator = partial(single_iteration_source_location,source_locations, direction, frequency_range,
                                     sky_param, noise_param, beam_param, calibration_scheme, save_to_disk,red_baseline_table, gain_table, amp_matrix,
                                     phase_matrix, red_tiles, red_groups, iterations)

    pool.map(iterator, numpy.arange(iterations))



    # Calculate run time
    endtime = time.time()
    runtime = endtime - starttime

    # Save input parameters to log file
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
    file.write("Runtime: " + str(runtime) + "\n")
    file.close()

    print "Runtime", runtime
    return


def single_iteration_source_location(source_locations, direction, frequency_range,
                                     sky_param, noise_param, beam_param, calibration_scheme, save_to_disk,red_baseline_table, gain_table, amp_matrix,
                                     phase_matrix, red_tiles, red_groups,n_iterations, iteration):


    n_tiles = len(red_tiles)
    n_groups = len(red_groups)
    n_locations = len(source_locations)
    noisy_amp_solutions = numpy.zeros((n_tiles + n_groups, n_locations))
    noisy_phase_solutions = numpy.zeros((n_tiles + n_groups, n_locations))

    ideal_amp_solutions = numpy.zeros((n_tiles + n_groups, n_locations))
    ideal_phase_solutions = numpy.zeros((n_tiles + n_groups, n_locations))

    if sky_param[0] == "background" or sky_param[0] == 'point_and_background':
        # Create the visibilities for the static background sky
        sky_model = ['background']
        obs_visibilities, ideal_visibilities, model_visibilities = \
            numerical_visibilities(red_baseline_table, frequency_range,
                                   [False], sky_model, beam_param, iteration)

    for location_index in range(len(source_locations)):
        if direction == 'l':
            l = source_locations[location_index]
            m = 0

        elif direction == 'm':
            l = 0
            m = source_locations[location_index]
        # add a point source (with noise) to background
        if sky_param[0] == 'point_and_background':
            sky_model = ['point', sky_param[1], l, m]
            if noise_param[0] and len(noise_param) == 4:
                noise_param[0] = 'SEFD'
            elif noise_param[0] and len(noise_param) == 1:
                noise_param[0] = 'source'
            point_obs_visibilities, point_ideal_visibilities, point_model_visibilities = \
                analytic_visibilities(red_baseline_table, frequency_range
                                       , noise_param, sky_model, beam_param, iteration)

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
                analytic_visibilities(red_baseline_table, frequency_range,
                                       noise_param, sky_model, beam_param, iteration)

            obs_visibilities += point_obs_visibilities
            ideal_visibilities += point_ideal_visibilities
            model_visibilities += point_model_visibilities

        # Create point source data in the absence of background sky data
        elif sky_param[0] == 'point':
            sky_model = ['point', sky_param[1], l, m]
            noise_param[0] = 'source'
            obs_visibilities, ideal_visibilities, model_visibilities = \
                analytic_visibilities(red_baseline_table, frequency_range
                                       , noise_param, sky_model, beam_param, iteration)

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

        ideal_amp_solutions[:, location_index] = ideal_amp_data
        ideal_phase_solutions[:, location_index] = ideal_phase_data

        # Use the noisy data  to solve for the antenna gains
        noisy_amp_data, noisy_phase_data = Redundant_Calibrator(
            amp_matrix, phase_matrix, obs_visibilities,
            red_baseline_table, red_tiles, red_groups, calibration_param)

        noisy_amp_solutions[:, location_index] = noisy_amp_data
        noisy_phase_solutions[:, location_index] = noisy_phase_data


        # remove the point source
        if sky_param[0] == 'point_and_background':
            obs_visibilities -= point_obs_visibilities
            ideal_visibilities -= point_ideal_visibilities
            model_visibilities -= point_model_visibilities


    parameters = numpy.concatenate((red_tiles, red_groups))
    axesdata = [parameters, source_locations]
    axeslabels = ['parameters', 'source_locations']
    output_types = ["ideal_amp", "ideal_phase", "noisy_amp", "noisy_phase"]


    prefix = str(0) * (len(str(n_iterations)) - len(str(iteration))) + str(iteration)

    save_to_hdf5(save_to_disk[1] + "threaded_" + output_types[0] + "/", prefix + "_SLPO_ideal_amp_solutions", ideal_amp_solutions,
             axesdata, axeslabels)
    save_to_hdf5(save_to_disk[1] + "threaded_" + output_types[1] + "/", prefix + "_SLPO_ideal_amp_solutions", ideal_phase_solutions,
             axesdata, axeslabels)

    save_to_hdf5(save_to_disk[1] + "threaded_" + output_types[2] + "/", prefix + "_SLPO_ideal_amp_solutions", noisy_amp_solutions,
             axesdata, axeslabels)
    save_to_hdf5(save_to_disk[1] + "threaded_" + output_types[3] + "/", prefix + "_SLPO_ideal_amp_solutions", noisy_phase_solutions,
             axesdata, axeslabels)