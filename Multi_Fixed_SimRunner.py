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


def source_flux_and_position_offset_changer_FixedMP(telescope_param, calibration_channel, noise_param, sky_param,
                                                    beam_param,
                                                    calibration_scheme, peakflux_range, offset_range, n_iterations,
                                                    save_to_disk,
                                                    processes):
    """
    """

    print "Simulating the Calibration of Arrays with Redundancy"
    print "Changing Maximum Flux and Position offsets"
    print "Fixed position offsets"
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

    xy_offsets = numpy.random.normal(0, 1, xyz_positions[:, 1:3].shape)


    numpy.savetxt(save_to_disk[1]+"position_offsets.txt", xy_offsets)

    file = open(save_to_disk[1] + "SFPO_simulation_parameters.log", "w")
    file.write("Changing Source Flux and Position Offset simulation\n")
    file.write("Fixed and Scaled Positions offsets\n")
    file.write("Telescope Parameters: " + str(telescope_param) + "\n")
    file.write("Calibration Channel: " + str(frequency_range / 1e6) + "MHz \n")
    file.write("Noise Parameters: " + str(noise_param) + "\n")
    file.write("Sky Model: " + str(sky_param) + "\n")
    file.write("Beam Parameters: " + str(beam_param) + "\n")
    file.write("Calibration scheme: " + str(calibration_scheme) + "\n")
    file.write("Offset Range: " + str(offset_range) + "\n")
    file.write("Peak Flux Range: " + str(peakflux_range) + "\n")
    file.write("Iterations: " + str(n_iterations) + "\n")
    file.close()


    pool = multiprocessing.Pool(processes=processes)
    iterator = partial(single_iteration_source_flux_position_offset_Fixed,
                       xyz_positions, gain_table, frequency_range, peak_fluxes, position_offsets, calibration_scheme,
                       sky_param, noise_param, beam_param, save_to_disk, red_tiles, red_groups, n_iterations, xy_offsets)

    pool.map(iterator, iterations)
    end_time = time.time()

    runtime = end_time - start_time
    print "Runtime", runtime
    file = open(save_to_disk[1] + "SFPO_simulation_parameters.log", "a")
    file.write("Runtime: " + str(runtime) + "\n")
    file.close()
    return


def single_iteration_source_flux_position_offset_Fixed(xyz_positions, gain_table, frequency_range, peak_fluxes,
                                                       position_offsets, calibration_scheme, sky_param,
                                                       noise_param,
                                                       beam_param, save_to_disk, red_tiles, red_groups,
                                                       n_processes, xy_offsets, iteration):
    parameters = numpy.concatenate((red_tiles, red_groups))

    noisy_amp_solutions = numpy.zeros((len(parameters), len(position_offsets), len(peak_fluxes)))
    noisy_phase_solutions = numpy.zeros((len(parameters), len(position_offsets), len(peak_fluxes)))
    ideal_amp_solutions = numpy.zeros((len(parameters), len(position_offsets), len(peak_fluxes)))
    ideal_phase_solutions = numpy.zeros((len(parameters), len(position_offsets), len(peak_fluxes)))

    for offset_index in range(len(position_offsets)):
        offset_positions = xyz_positions.copy()
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

    prefix = str(0) * (len(str(n_processes)) - len(str(iteration))) + str(iteration)

    save_to_hdf5(save_to_disk[1] + "threaded_" + output_types[0] + "/", prefix + "_SFPO_ideal_amp_solutions",
                 ideal_amp_solutions, axesdata, axeskeys)
    save_to_hdf5(save_to_disk[1] + "threaded_" + output_types[1] + "/", prefix + "_SFPO_ideal_phase_solutions",
                 ideal_phase_solutions, axesdata, axeskeys)
    save_to_hdf5(save_to_disk[1] + "threaded_" + output_types[2] + "/", prefix + "_SFPO_noisy_amp_solutions",
                 noisy_amp_solutions, axesdata, axeskeys)
    save_to_hdf5(save_to_disk[1] + "threaded_" + output_types[3] + "/", prefix + "_SFPO_noisy_phase_solutions",
                 noisy_phase_solutions, axesdata, axeskeys)

    return


def source_location_and_position_offset_changer_FixedMP(telescope_param, calibration_channel, noise_param, sky_param,
                                                        beam_param,
                                                        calibration_scheme, source_position_range, offset_range,
                                                        n_iterations,
                                                        save_to_disk,
                                                        processes):
    """
    """
    print "Simulating the Calibration of Arrays with Redundancy"
    print "Changing the source location and Position offsets"
    print "Fixed position offsets"
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




    print "current setting l_steps=:",source_position_range[2]
    max_b = numpy.max(numpy.abs(baseline_table[:, 2:4, -1]))
    min_l = 1. / max_b
    delta_l = 0.1 * min_l
    n_l_steps = int((source_position_range[1]-source_position_range[0]) / delta_l)

    if n_l_steps >= source_position_range[2]:
        source_position_range[2] = n_l_steps
        print "Warning: source location was too low, increased to", n_l_steps
    elif n_l_steps >= 999:
        source_position_range[2] = 999

    source_locations = numpy.linspace(source_position_range[0], source_position_range[1], source_position_range[2])

    file = open(save_to_disk[1] + "SLPO_simulation_parameters.log", "w")
    file.write("Changing Source Location and Position Offset simulation\n")
    file.write("Fixed and Scaled Positions offsets\n")
    file.write("Telescope Parameters: " + str(telescope_param) + "\n")
    file.write("Calibration Channel: " + str(frequency_range / 1e6) + "MHz \n")
    file.write("Noise Parameters: " + str(noise_param) + "\n")
    file.write("Sky Model: " + str(sky_param) + "\n")
    file.write("Source location parameters: " + str(source_position_range) + "\n")
    file.write("Calibration scheme: " + str(calibration_scheme) + "\n")
    file.write("Offset Range: " + str(offset_range) + "\n")
    file.write("Beam Parameters: " + str(beam_param) + "\n")
    file.write("Iterations: " + str(n_iterations) + "\n")
    file.close()


    xy_offsets = numpy.random.normal(0, 1, xyz_positions[:, 1:3].shape)
    numpy.savetxt(save_to_disk[1]+"position_offsets.txt", xy_offsets)

    pool = multiprocessing.Pool(processes=processes)
    iterator = partial(single_iteration_source_location_position_offset_Fixed,
                       xyz_positions, gain_table, frequency_range, source_locations, position_offsets,
                       calibration_scheme, sky_param, noise_param, beam_param, save_to_disk, red_tiles, red_groups,
                       n_iterations, xy_offsets)

    pool.map(iterator, iterations)
    end_time = time.time()

    runtime = end_time - start_time
    print "Runtime", runtime
    file = open(save_to_disk[1] + "SLPO_simulation_parameters.log", "a")
    file.write("Runtime: " + str(runtime) + "\n")
    file.close()
    return


def single_iteration_source_location_position_offset_Fixed(xyz_positions, gain_table, frequency_range,
                                                           source_locations, position_offsets,
                                                           calibration_scheme, sky_param, noise_param,
                                                           beam_param, save_to_disk, red_tiles, red_groups, n_processes,
                                                           xy_offsets, iteration):
    parameters = numpy.concatenate((red_tiles, red_groups))

    noisy_amp_solutions = numpy.zeros((len(parameters), len(position_offsets), len(source_locations)))
    noisy_phase_solutions = numpy.zeros((len(parameters), len(position_offsets), len(source_locations)))
    ideal_amp_solutions = numpy.zeros((len(parameters), len(position_offsets), len(source_locations)))
    ideal_phase_solutions = numpy.zeros((len(parameters), len(position_offsets), len(source_locations)))

    for offset_index in range(len(position_offsets)):
        offset_positions = xyz_positions.copy()
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

    prefix = str(0) * (len(str(n_processes)) - len(str(iteration))) + str(iteration)

    save_to_hdf5(save_to_disk[1] + "threaded_" + output_types[0] + "/", prefix + "_SLPO_ideal_amp_solutions",
                 ideal_amp_solutions, axesdata, axeskeys)
    save_to_hdf5(save_to_disk[1] + "threaded_" + output_types[1] + "/", prefix + "_SLPO_ideal_phase_solutions",
                 ideal_phase_solutions, axesdata, axeskeys)
    save_to_hdf5(save_to_disk[1] + "threaded_" + output_types[2] + "/", prefix + "_SLPO_noisy_amp_solutions",
                 noisy_amp_solutions, axesdata, axeskeys)
    save_to_hdf5(save_to_disk[1] + "threaded_" + output_types[3] + "/", prefix + "_SLPO_noisy_phase_solutions",
                 noisy_phase_solutions, axesdata, axeskeys)

    return
