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
from GeneralTools import unique_value_finder
from GeneralTools import position_finder
from GeneralTools import solution_averager
from GeneralTools import FourD_solution_averager
from GeneralTools import visibility_histogram_plotter
from GeneralTools import solution_histogram_plotter
from GeneralTools import TrueSolutions_Organizer
from GeneralTools import save_to_hdf5
from Gridded_Skymodel import CreateVisibilities
from RedundantCalibration import Redundant_Calibrator
from RedundantCalibration import LogcalMatrixPopulator


def source_flux_and_position_offset_changer_parallel(telescope_param, calibration_channel, noise_param, sky_param, beam_param,
                                            calibration_scheme, peakflux_range, offset_range, n_iterations,
                                            save_to_disk,
                                            processes):
    """
    """

    print "Simulating the Calibration of Arrays with Redundancy"
    print "Changing Maximum Flux and Position offsets"
    start_time = time.time()

    minimum_position_offset = numpy.log10(offset_range[0])
    maximum_position_offset = numpy.log10(offset_range[1])
    position_step_number = offset_range[2]
    position_offsets = numpy.logspace(minimum_position_offset, maximum_position_offset, position_step_number)

    minimum_peakflux = numpy.log10(peakflux_range[0])
    maximum_peakflux = numpy.log10(peakflux_range[1])
    peakflux_step_number = peakflux_range[2]
    peak_fluxes = numpy.logspace(minimum_peakflux, maximum_peakflux, peakflux_step_number)

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

    # Knowing what we're solving for we can start setting up tables


    pool = multiprocessing.Pool(processes=processes)
    iterator = partial(single_iteration_source_flux_position_offset,xyz_positions, gain_table, frequency_range, peak_fluxes,
                                                 position_offsets, calibration_scheme, sky_param, noise_param,
                                                 beam_param, save_to_disk, red_baseline_table, red_tiles, red_groups,
                                                 n_iterations)

    pool.map(iterator, numpy.arange(n_iterations))
    end_time = time.time()

    runtime = end_time - start_time
    print "Runtime", runtime
    file = open(save_to_disk[1] + "SFPO_simulation.log", "w")
    file.write("Moving source and changing position offset simulation")
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
                                                 position_offsets, calibration_scheme, sky_param, noise_param,
                                                 beam_param, save_to_disk, red_baseline_table, red_tiles, red_groups,
                                                 n_iterations, iteration):
    # Have a fixed seed
    numpy.random.seed(iteration)


    n_tiles = len(red_tiles)
    n_groups = len(red_groups)
    n_peakfluxes = len(peak_fluxes)
    n_offsets = len(position_offsets)
    noisy_amp_solutions = numpy.zeros((n_tiles + n_groups, n_peakfluxes, n_offsets))
    noisy_phase_solutions = numpy.zeros((n_tiles + n_groups, n_peakfluxes, n_offsets))
    ideal_amp_solutions = numpy.zeros((n_tiles + n_groups, n_peakfluxes, n_offsets))
    ideal_phase_solutions = numpy.zeros((n_tiles + n_groups, n_peakfluxes, n_offsets))

    sigma_counter = 0
    for sigma in position_offsets:

        # We want to generate an array which has offset but is still completely redundant!
        array_counter = 0
        while True:
            # Generate positions offsets to add to the antenna positions
            x_offset = numpy.random.normal(0, sigma, gain_table[:, 1].shape[0])
            y_offset = numpy.random.normal(0, sigma, gain_table[:, 2].shape[0])

            offset_positions = xyz_positions.copy()
            offset_positions[:, 1] += x_offset
            offset_positions[:, 2] += y_offset

            offset_baseline_table = baseline_converter(offset_positions, gain_table, frequency_range, verbose=False)
            off_red_baseline_table = redundant_baseline_finder(offset_baseline_table, 'ALL')

            if off_red_baseline_table.shape == red_baseline_table.shape:
                amp_matrix, phase_matrix, off_red_tiles, off_red_groups = LogcalMatrixPopulator(off_red_baseline_table,
                                                                                                offset_positions)
                # now we check whether the number of redundant tiles and groups is still the same
                if len(off_red_tiles) == n_tiles and len(off_red_groups) == n_groups:
                    array_succes = True
                    break
                elif array_counter > 100:
                    array_succes = False
                    break
            else:
                array_counter += 1

        if sky_param[0] == 'point_and_background':
            # Create the visibilities for the static background sky
            sky_model = ['background']
            background_noise = [False]
            background_obs_visibilities, background_ideal_visibilities, background_model_visibilities = \
                CreateVisibilities(off_red_baseline_table, frequency_range, background_noise, sky_model, beam_param,
                                       iteration)
        elif sky_param[0] == "background":
            # Create the visibilities for the static background sky
            sky_model = ['background']
            background_noise = [False]
            obs_visibilities, ideal_visibilities, model_visibilities = \
                CreateVisibilities(off_red_baseline_table, frequency_range, background_noise, sky_model, beam_param,
                                       iteration)

        peakflux_counter = 0
        for peakflux in peak_fluxes:
            # If we managed to create redundant telescope
            # print array_succes
            if array_succes:
                # add a point source (with noise) to background

                if sky_param[0] == 'point_and_background':
                    pointsky_model = ['point', peakflux, sky_param[2], sky_param[3]]
                    if noise_param[0] and len(noise_param) == 4:
                        noise_param[0] = 'SEFD'
                    elif noise_param[0] and len(noise_param) == 1:
                        noise_param[0] = 'source'
                    point_obs_visibilities, point_ideal_visibilities, point_model_visibilities = \
                        CreateVisibilities(off_red_baseline_table, frequency_range, noise_param, pointsky_model, beam_param,
                                       iteration)

                    obs_visibilities = point_obs_visibilities + background_obs_visibilities
                    ideal_visibilities = point_ideal_visibilities + background_ideal_visibilities
                    model_visibilities = point_model_visibilities + background_model_visibilities


                # Create point source data in the absence of background sky data
                elif sky_param[0] == 'point':
                    pointsky_model = ['point', peakflux, sky_param[2], sky_param[3]]
                    noise_param[0] = 'source'
                    obs_visibilities, ideal_visibilities, model_visibilities = \
                        CreateVisibilities(off_red_baseline_table, frequency_range, noise_param, pointsky_model, beam_param,
                                       iteration)

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

                noisy_amp_solutions[:, peakflux_counter, sigma_counter] = noisy_amp_results
                noisy_phase_solutions[:, peakflux_counter, sigma_counter] = noisy_phase_results
                ideal_amp_solutions[:, peakflux_counter, sigma_counter] = ideal_amp_results
                ideal_phase_solutions[:, peakflux_counter, sigma_counter] = ideal_phase_results
            else:
                noisy_amp_solutions[:, peakflux_counter, sigma_counter] = numpy.nan
                noisy_phase_solutions[:, peakflux_counter, sigma_counter] = numpy.nan
                ideal_amp_solutions[:, peakflux_counter, sigma_counter] = numpy.nan
                ideal_phase_solutions[:, peakflux_counter, sigma_counter] = numpy.nan

            peakflux_counter += 1
        sigma_counter += 1
    parameters = numpy.concatenate((red_tiles, red_groups))
    axesdata = [parameters, position_offsets, peak_fluxes]
    axeskeys = ['parameters', 'positions_uncertainty', 'peak_fluxes']
    output_types = ["ideal_amp", "ideal_phase", "noisy_amp", "noisy_phase"]

    prefix = str(0) * (len(str(n_iterations)) - len(str(iteration))) + str(iteration)
    save_to_hdf5(save_to_disk[1] + "threaded_" + output_types[0] + "/", prefix + "_SFPO_ideal_amp_solutions", ideal_amp_solutions,
                     axesdata, axeskeys)
    save_to_hdf5(save_to_disk[1] + "threaded_" + output_types[1] + "/", prefix + "_SFPO_ideal_phase_solutions", ideal_phase_solutions,
                     axesdata, axeskeys)

    save_to_hdf5(save_to_disk[1] + "threaded_" + output_types[2] + "/", prefix + "_SFPO_noisy_amp_solutions", noisy_amp_solutions,
                     axesdata, axeskeys)
    save_to_hdf5(save_to_disk[1] + "threaded_" + output_types[3] + "/", prefix + "_SFPO_noisy_phase_solutions", noisy_phase_solutions,
                     axesdata, axeskeys)

    return


def source_location_and_position_offset_changer_parallel(telescope_param, calibration_channel, noise_param, sky_param, beam_param,
                                            calibration_scheme, offset_range, n_iterations,
                                            save_to_disk,
                                            processes):
    """
    """

    print "Simulating the Calibration of Arrays with Redundancy"
    print "Changing Maximum Flux and Position offsets"
    start_time = time.time()

    minimum_position_offset = numpy.log10(offset_range[0])
    maximum_position_offset = numpy.log10(offset_range[1])
    position_step_number = offset_range[2]
    position_offsets = numpy.logspace(minimum_position_offset, maximum_position_offset, position_step_number)

    source_positions = numpy.linspace(-1, 1, 5)

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

    # Knowing what we're solving for we can start setting up tables


    pool = multiprocessing.Pool(processes=processes)
    iterator = partial(single_iteration_source_location_position_offset, xyz_positions, gain_table, frequency_range,
                       source_positions,
                                                 position_offsets, calibration_scheme, sky_param, noise_param,
                                                 beam_param, save_to_disk, red_baseline_table, red_tiles, red_groups,
                                                 n_iterations)

    pool.map(iterator, numpy.arange(n_iterations))
    end_time = time.time()

    runtime = end_time - start_time
    print "Runtime", runtime
    file = open(save_to_disk[1] + "SFPO_simulation.log", "w")
    file.write("Moving source and changing position offset simulation")
    file.write("Telescope Parameters: " + str(telescope_param) + "\n")
    file.write("Calibration Channel: " + str(frequency_range / 1e6) + "MHz \n")
    file.write("Noise Parameters: " + str(noise_param) + "\n")
    file.write("Sky Model: " + str(sky_param) + "\n")
    file.write("Beam Parameters: " + str(beam_param) + "\n")
    file.write("Calibration scheme: " + str(calibration_scheme) + "\n")
    file.write("Offset Range: " + str(offset_range) + "\n")
    file.write("Iterations: " + str(n_iterations) + "\n")
    file.write("Runtime: " + str(runtime) + "\n")
    file.close()
    return


def single_iteration_source_location_position_offset(xyz_positions, gain_table, frequency_range, source_positions,
                                                     position_offsets, calibration_scheme, sky_param, noise_param,
                                                     beam_param, save_to_disk, red_baseline_table, red_tiles, red_groups,
                                                     n_iterations, iteration):
    # Have a fixed seed
    numpy.random.seed(iteration)


    n_tiles = len(red_tiles)
    n_groups = len(red_groups)
    n_locations = len(source_positions)
    n_offsets = len(position_offsets)
    noisy_amp_solutions = numpy.zeros((n_tiles + n_groups, n_locations, n_offsets))
    noisy_phase_solutions = numpy.zeros((n_tiles + n_groups, n_locations, n_offsets))
    ideal_amp_solutions = numpy.zeros((n_tiles + n_groups, n_locations, n_offsets))
    ideal_phase_solutions = numpy.zeros((n_tiles + n_groups, n_locations, n_offsets))

    sigma_counter = 0
    for sigma in position_offsets:

        # We want to generate an array which has offset but is still completely redundant!
        array_counter = 0
        while True:
            # Generate positions offsets to add to the antenna positions
            x_offset = numpy.random.normal(0, sigma, gain_table[:, 1].shape[0])
            y_offset = numpy.random.normal(0, sigma, gain_table[:, 2].shape[0])

            offset_positions = xyz_positions.copy()
            offset_positions[:, 1] += x_offset
            offset_positions[:, 2] += y_offset

            offset_baseline_table = baseline_converter(offset_positions, gain_table, frequency_range, verbose=False)
            off_red_baseline_table = redundant_baseline_finder(offset_baseline_table, 'ALL')

            if off_red_baseline_table.shape == red_baseline_table.shape:
                amp_matrix, phase_matrix, off_red_tiles, off_red_groups = LogcalMatrixPopulator(off_red_baseline_table,
                                                                                                offset_positions)
                # now we check whether the number of redundant tiles and groups is still the same
                if len(off_red_tiles) == n_tiles and len(off_red_groups) == n_groups:
                    array_succes = True
                    break
                elif array_counter > 100:
                    array_succes = False
                    break
            else:
                array_counter += 1

        if sky_param[0] == "background" or sky_param[0] == 'point_and_background':
            # Create the visibilities for the static background sky
            sky_model = ['background']
            background_noise = [False]
            background_obs_visibilities, background_ideal_visibilities, background_model_visibilities = \
                CreateVisibilities(off_red_baseline_table, frequency_range, background_noise, sky_model, beam_param,
                                       iteration)
        location_counter = 0
        print source_positions
        for source_location in source_positions:
            # If we managed to create redundant telescope
            # print array_succes
            if array_succes:
                # add a point source (with noise) to background
                if sky_param[0] == 'point_and_background':
                    sky_model = ['point', sky_param[1], source_location, sky_param[3]]
                    if noise_param[0] and len(noise_param) == 4:
                        noise_param[0] = 'SEFD'
                    elif noise_param[0] and len(noise_param) == 1:
                        noise_param[0] = 'source'
                    point_obs_visibilities, point_ideal_visibilities, point_model_visibilities = \
                        CreateVisibilities(off_red_baseline_table, frequency_range, noise_param, sky_model, beam_param,
                                       iteration)

                    obs_visibilities = point_obs_visibilities + background_obs_visibilities
                    ideal_visibilities = point_ideal_visibilities + background_ideal_visibilities
                    model_visibilities = point_model_visibilities + background_model_visibilities


                # Create point source data in the absence of background sky data
                elif sky_param[0] == 'point':
                    sky_model = ['point', sky_param[1], source_location, sky_param[3]]
                    noise_param[0] = 'source'
                    obs_visibilities, ideal_visibilities, model_visibilities = \
                        CreateVisibilities(off_red_baseline_table, frequency_range, noise_param, sky_model, beam_param,
                                       iteration)

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

                noisy_amp_solutions[:, location_counter, sigma_counter] = noisy_amp_results
                noisy_phase_solutions[:, location_counter, sigma_counter] = noisy_phase_results
                ideal_amp_solutions[:, location_counter, sigma_counter] = ideal_amp_results
                ideal_phase_solutions[:, location_counter, sigma_counter] = ideal_phase_results
            else:
                noisy_amp_solutions[:, location_counter, sigma_counter] = numpy.nan
                noisy_phase_solutions[:, location_counter, sigma_counter] = numpy.nan
                ideal_amp_solutions[:, location_counter, sigma_counter] = numpy.nan
                ideal_phase_solutions[:, location_counter, sigma_counter] = numpy.nan


            if sky_param[0] == 'point_and_background':
                obs_visibilities -= point_obs_visibilities
                ideal_visibilities -= point_ideal_visibilities
                model_visibilities -= point_model_visibilities
            location_counter += 1
        sigma_counter += 1

    parameters = numpy.concatenate((red_tiles, red_groups))
    axesdata = [parameters, position_offsets, source_positions]
    axeskeys = ['parameters', 'positions_uncertainty', 'peak_fluxes']
    output_types = ["ideal_amp", "ideal_phase", "noisy_amp", "noisy_phase"]

    prefix = str(0) * (len(str(n_iterations)) - len(str(iteration))) + str(iteration)
    save_to_hdf5(save_to_disk[1] + "threaded_" + output_types[0] + "/", prefix + "_SFPO_ideal_amp_solutions", ideal_amp_solutions,
                     axesdata, axeskeys)
    save_to_hdf5(save_to_disk[1] + "threaded_" + output_types[1] + "/", prefix + "_SFPO_ideal_phase_solutions", ideal_phase_solutions,
                     axesdata, axeskeys)

    save_to_hdf5(save_to_disk[1] + "threaded_" + output_types[2] + "/", prefix + "_SFPO_noisy_amp_solutions", noisy_amp_solutions,
                     axesdata, axeskeys)
    save_to_hdf5(save_to_disk[1] + "threaded_" + output_types[3] + "/", prefix + "_SFPO_noisy_phase_solutions", noisy_phase_solutions,
                     axesdata, axeskeys)

    return
