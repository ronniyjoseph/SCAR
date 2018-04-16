import numpy
import os
import sys
import time
from multiprocessing import Pool
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


def single_iteration_source_flux_