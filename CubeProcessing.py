import numpy
import h5py
import os
import subprocess
import sys
from GeneralTools import table_setup
from matplotlib import pyplot
from matplotlib import rcParams
from matplotlib import rc
from GeneralTools import save_to_hdf5

rcParams['mathtext.default'] = 'regular'
rcParams.update({'figure.autolayout': True})

"""
This module contains all functions required to process the full solutions hdf5 files
*solution histogram plotter makes videos of the cube as the sources moves along the sky
*Solution averager calculates the mean/median and std/iqr
"""



def data_processor(output_path, simulation_type, stacking_mode, histogram_plotset, averaging_param):
    if simulation_type == "CRAMPS":
        if stacking_mode[0]:
            data_stacker3D(output_path, simulation_type)
        if histogram_plotset[0]:
            CRAMPS_histogram_inspection(output_path + simulation_run, histogram_plotset)
        elif averaging_param[0]:
            solution_averager(output_path, averaging_param, "ideal", "amp")
            solution_averager(output_path, averaging_param, "ideal", "phase")

            solution_averager(output_path, averaging_param, "noisy", "amp")
            solution_averager(output_path, averaging_param, "noisy", "phase")
        else:
            sys.exit("blaah")
    elif simulation_type == "SFPO" or simulation_type == "SLPO":
        if stacking_mode[0]:
            data_stacker4D(output_path, simulation_type)

        if histogram_plotset[0]:
            SiSps_histogram_inspection(output_path, simulation_type, histogram_plotset[1])
        else:
            sys.exit("blaah")
    else:
        sys.exit("Simulation type unknown: Please choose 'CRAMPS' or 'SiSpS'")
    return

def data_stacker3D(folder, simulation_type):
    output_list = ["ideal_amp", "ideal_phase", "noisy_amp", "noisy_phase"]
    for output in output_list:
        thread_path = folder + "/threaded_" + output
        list_directory = sorted(os.listdir(thread_path))
        test_index = 0
        # open op a file to get the right dimensions
        solution_slice = h5py.File(thread_path + "/" + list_directory[test_index], 'r')
        axes_keys = solution_slice.keys()
        solution_data = solution_slice['data'][:]
        solution_axes1 = solution_slice[axes_keys[1]][:]
        solution_axes2 = solution_slice[axes_keys[2]][:]
        solution_slice.close()

        print ""
        print "Input stuff"

        data_cube = numpy.zeros(
            (solution_data.shape[0], solution_data.shape[1], len(list_directory)))
        counter = 0
        for file_name in list_directory:
            solution_slice = h5py.File(thread_path + "/" + file_name, 'r')
            data_cube[:, :, counter] = solution_slice['data']
            solution_slice.close()
            counter += 1
        data_axes = [solution_axes1, solution_axes2]
        axes_keys[2] = 'source_locations'
        cube_name = output + "_solutions"
        save_to_hdf5(folder, cube_name, data_cube, data_axes, axes_keys[1:])



def data_stacker4D(folder, simulation_type):
    output_list = ["ideal_amp", "ideal_phase", "noisy_amp", "noisy_phase"]
    for output in output_list:
        thread_path = folder + "/threaded_" + output
        list_directory = sorted(os.listdir(thread_path))
        print len(list_directory)
        test_index = 0
        # open op a file to get the right dimensions
        solution_slice = h5py.File(thread_path + "/" + list_directory[test_index], 'r')
        print thread_path + "/" + list_directory[test_index]
        axes_keys = solution_slice.keys()

        solution_data = solution_slice['data'][:]
        solution_axes1 = solution_slice[axes_keys[1]][:]
        solution_axes2 = solution_slice[axes_keys[2]][:]
        solution_axes3 = solution_slice[axes_keys[3]][:]
        solution_slice.close()
        print ""
        print "Input stuff"

        data_cube = numpy.zeros(
            (solution_data.shape[0], solution_data.shape[1], solution_data.shape[2], len(list_directory)))
        counter = 0
        for file_name in list_directory:
            solution_slice = h5py.File(thread_path + "/" + file_name, 'r')
            data_cube[:, :, :, counter] = solution_slice['data']
            solution_slice.close()
            counter += 1
        data_axes = [solution_axes1, solution_axes2, solution_axes3, numpy.arange(len(list_directory))]
        axes_keys.append("iteration")
        cube_name = simulation_type + "_" + output + "_solutions"
        print cube_name
        save_to_hdf5(folder, cube_name, data_cube, data_axes, axes_keys[1:])




def SiSps_histogram_inspection(output_folder, simulation_type, solution_type):
    if solution_type == "ideal":
        create_solution_histogram_tile(output_folder, simulation_type, "ideal", "amp")
        create_solution_histogram_tile(output_folder, simulation_type, "ideal", "phase")
    elif solution_type == "noisy":
        create_solution_histogram_tile(output_folder, simulation_type, "noisy", "amp")
        create_solution_histogram_tile(output_folder, simulation_type, "noisy", "phase")
    elif solution_type == "both":
        create_solution_histogram_tile(output_folder, simulation_type, "both", "amp")
        create_solution_histogram_tile(output_folder, simulation_type, "both", "phase")
    else:
        sys.exit("solution_type: please select 'ideal','noisy' or 'both', Goodbye.")
    return


def create_solution_histogram_tile(output_folder, simulation_type, solution_type, solution_parameter):
    print ""
    print "Loading data"
    if solution_type == "ideal":
        ideal_solution_data, parameters, position_offsets, peak_fluxes, ideal_iterations = \
            SiSpS_cube_loader(output_folder, simulation_type, "ideal", solution_parameter)

        # indices = numpy.where(abs(ideal_solution_quantity) > 5e7)[0]
        # number_visibilities = len(indices)
        # number_tiles = len(ideal_solution_quantity) - number_visibilities

    elif solution_type == "noisy":
        noisy_solution_data, parameters, position_offsets, peak_fluxes, noisy_iterations = \
            SiSpS_cube_loader(output_folder, simulation_type, "noisy", solution_parameter)

        # indices = numpy.where(abs(noisy_solution_quantity) > 5e7)[0]
        # number_visibilities = len(indices)
        # number_tiles = len(noisy_solution_quantity) - number_visibilities

    elif solution_type == "both":
        ideal_solution_data, ideal_solution_quantity, ideal_position_offsets, ideal_peak_fluxes, ideal_iterations = \
            SiSpS_cube_loader(output_folder, simulation_type, "ideal", solution_parameter)
        noisy_solution_data, noisy_solution_quantity, noisy_position_offsets, noisy_peak_fluxes, noisy_iterations = \
            SiSpS_cube_loader(output_folder, simulation_type, "noisy", solution_parameter)
        ideal_indices = numpy.where(abs(ideal_solution_quantity) > 5e7)[0]
        noisy_indices = numpy.where(abs(ideal_solution_quantity) > 5e7)[0]
        if len(ideal_indices) != len(noisy_indices):
            sys.exit("Number of redundant visibilities differs in ideal and noisy files, something is wrong")
        else:
            parameters = ideal_solution_quantity
            position_offsets = ideal_position_offsets
            peak_fluxes = ideal_peak_fluxes
            number_visibilities = len(ideal_indices)
            # number_tiles = len(ideal_solution_quantity) - number_visibilities
    else:
        sys.exit("Invalid solution type: 'ideal', 'noisy' or 'both'")

    loop_number = 0
    while True:
        if loop_number == 0:
            print "Welcome to the interactive SiSps HDF5 Cube Histogram Plotter, which quantity do you want to plot?"
        else:
            print ""
            print "Do you want inspect another quantity:"

        for quantity in parameters:
            if quantity < 5e7:
                print "Antenna: ", int(quantity)
            else:
                print "Visibility: ", int(quantity)
        print "To exit : q"
        user_choice = raw_input("Your choice of the day :")

        if user_choice == "q":
            subprocess.call("clear", shell=True)
            break
        else:
            user_quantity = int(user_choice)
            quantity_index = numpy.where(user_quantity == parameters.astype(int))[0]

            # pass this along to plotting function which creates a tile
            print "Generating plots"
            fig1 = pyplot.figure(figsize=(4 * len(position_offsets), 4 * len(peak_fluxes)))
            if solution_type == "ideal":
                fig1 = plot_solution_histogram_tile(fig1, parameters[quantity_index],
                                                    ideal_solution_data[quantity_index, :, :, :],
                                                    position_offsets, peak_fluxes)
            elif solution_type == "noisy":
                fig1 = plot_solution_histogram_tile(fig1, parameters[quantity_index],
                                                    noisy_solution_data[quantity_index, :, :, :],
                                                    position_offsets, peak_fluxes)
            elif solution_type == "both":
                fig1 = plot_solution_histogram_tile(fig1, parameters[quantity_index],
                                                    [noisy_solution_data[quantity_index, :, :, :],
                                                     ideal_solution_data[quantity_index, :, :, :]],
                                                    position_offsets, peak_fluxes, ["C2", "C1"])

            fig1.suptitle(r'' + str(parameters[quantity_index]) + 'solutions', y=1.001)
            pyplot.show()
        loop_number += 1
        subprocess.call("clear", shell=True)

    return


def plot_solution_histogram_tile(fig1, quantity_number, solution_data, position_offsets, peak_fluxes, color="b"):
    # General plot formatting tools
    labelfontsize = 14
    amplitude_plotscale = 'log'
    phase_plotscale = 'linear'
    number_bins = 100
    row_start =    0
    row_end =  len(position_offsets)

    col_start = 0
    col_end = len(peak_fluxes)

    stepsize = 2
    if (row_end - row_start) / stepsize > 5:
        rows = numpy.arange(row_start, row_end, (row_end - row_start) / 5)
        print rows
    else:
        rows = numpy.arange(row_start, row_end, stepsize)
    if (col_end - col_start) / stepsize > 5:
        cols = numpy.arange(col_start, col_end, (col_end - col_start) / 5)
    else:
        cols = numpy.arange(col_start, col_end, stepsize)

    nrow = len(rows)
    ncol = len(cols)
    plotcounter = 1

    #################
    ###################
    #################
    #solution_data = numpy.abs(solution_data)
    #######################
    ############

    for offset_index in rows:
        for flux_index in cols:
            subplot = fig1.add_subplot(nrow, ncol, plotcounter)

            if len(solution_data) == 1:
                histogram_data = solution_data[0, offset_index, flux_index, :]

            elif len(solution_data) > 1:
                histogram_data = []

                for dataset_number in range(len(solution_data)):

                    selected_data = solution_data[dataset_number][0,offset_index,flux_index, :]
                    print selected_data[::100]

                    histogram_data.append(selected_data[~numpy.isnan(selected_data)])


                bin_counts, _, _ = subplot.hist(histogram_data, histtype='stepfilled', edgecolor='none', alpha=0.4,
                                            bins=number_bins, color=color)
            subplot.text(0.95, 0.01, r'$log [\sigma] =%s$' % (
                str(numpy.around(numpy.log10(position_offsets[offset_index]), decimals=2))),
                         verticalalignment='bottom', horizontalalignment='right',
                         transform=subplot.transAxes, fontsize=labelfontsize)
            subplot.text(0.95, 0.21, r'$S =  %s Jy$' % (str(numpy.around(peak_fluxes[flux_index], decimals=2))),
                         verticalalignment='bottom', horizontalalignment='right',
                         transform=subplot.transAxes, fontsize=labelfontsize)

            subplot.set_yscale('log')

            minimum = numpy.min(bin_counts[0])

            maximum = numpy.max(bin_counts[0])

            #subplot.set_ylim([minimum, maximum])
            subplot.set_xlim([0,2])
            plotcounter += 1
    return fig1


def SiSpS_cube_loader(output_folder, simulation_type, solution_type, solution_parameter):

    solution_cube = h5py.File(
        output_folder + "/" + simulation_type + "_" + solution_type + "_" + solution_parameter + "_solutions.h5", 'r')
    print output_folder + "/" + simulation_type + "_" + solution_type + "_" + solution_parameter + "_solutions.h5"

    solution_data = solution_cube['data'][:]
    solution_quantity = solution_cube['parameters'][:]
    position_offsets = solution_cube['positions_uncertainty'][:]
    peak_fluxes = solution_cube['peak_fluxes'][:]
    iterations = solution_cube['iteration'][:]
    solution_cube.close()

    return solution_data, solution_quantity, position_offsets, peak_fluxes, iterations


def CRAMPS_histogram_inspection(output_folder, solution_type):
    if solution_type == "ideal":
        create_solution_histogram_video(output_folder, "ideal", "amp")
        create_solution_histogram_video(output_folder, "ideal", "phase")
    elif solution_type == "noisy":
        create_solution_histogram_video(output_folder, "noisy", "amp")
        create_solution_histogram_video(output_folder, "noisy", "phase")
    elif solution_type == "both":
        create_solution_histogram_video(output_folder, "both", "amp")
        create_solution_histogram_video(output_folder, "both", "phase")
    else:
        sys.exit("solution_type: please select 'ideal','noisy' or 'both', Goodbye.")
    return


def create_solution_histogram_video(output_folder, solution_type, solution_parameter):
    if not os.path.exists(output_folder + '/temp_' + solution_parameter + '/'):
        print ""
        print "Creating temporary output folder in " + output_folder + " directory"
        os.makedirs(output_folder + '/temp_' + solution_parameter + '/')

    print ""
    print "Loading data"
    if solution_type == "ideal":
        ideal_solution_data, ideal_solution_quantity, l, ideal_iterations = CRAMPS_cube_loader(output_folder,
                                                                                               solution_type,
                                                                                               solution_parameter)
        indices = numpy.where(abs(ideal_solution_quantity) > 5e7)[0]
        number_visibilities = len(indices)
        number_tiles = len(ideal_solution_quantity) - number_visibilities

    elif solution_type == "noisy":
        noisy_solution_data, noisy_solution_quantity, l, noisy_iterations = CRAMPS_cube_loader(output_folder,
                                                                                               solution_type,
                                                                                               solution_parameter)
        indices = numpy.where(abs(noisy_solution_quantity) > 5e7)[0]
        number_visibilities = len(indices)
        number_tiles = len(noisy_solution_quantity) - number_visibilities

    elif solution_type == "both":
        ideal_solution_data, ideal_solution_quantity, ideal_l, ideal_iterations = CRAMPS_cube_loader(output_folder,
                                                                                                     "ideal",
                                                                                                     solution_parameter)
        noisy_solution_data, noisy_solution_quantity, noisy_l, noisy_iterations = CRAMPS_cube_loader(output_folder,
                                                                                                     "noisy",
                                                                                                     solution_parameter)
        ideal_indices = numpy.where(abs(ideal_solution_quantity) > 5e7)[0]
        noisy_indices = numpy.where(abs(ideal_solution_quantity) > 5e7)[0]
        if len(ideal_indices) != len(noisy_indices):
            sys.exit("Number of redundant visibilities differs in ideal and noisy files, something is wrong")
        else:
            number_visibilities = len(ideal_indices)
            number_tiles = len(ideal_solution_quantity) - number_visibilities
            l = noisy_l

    print ""
    print "Creating histogram video of the solutions"

    print "Generating plots"
    for i in range(len(l)):
        fig1 = pyplot.figure(figsize=(16, 10))
        if solution_type == "ideal":
            fig1 = solution_histogram_plotter(fig1, number_visibilities, number_tiles, ideal_solution_data,
                                              solution_parameter, ideal_solution_quantity, i)
        elif solution_type == "noisy":
            fig1 = solution_histogram_plotter(fig1, number_visibilities, number_tiles, noisy_solution_data,
                                              solution_parameter, noisy_solution_quantity, i)
        elif solution_type == "both":
            fig1 = solution_histogram_plotter(fig1, number_visibilities, number_tiles, ideal_solution_data,
                                              solution_parameter, ideal_solution_quantity, i, "r")
            fig1 = solution_histogram_plotter(fig1, number_visibilities, number_tiles, noisy_solution_data,
                                              solution_parameter, noisy_solution_quantity, i, "b")

        fig1.suptitle(r'' + solution_parameter + 'solutions at $\mathit{l}$=' + str(l[i]), y=1.001)
        fig1.savefig(
            output_folder + '/temp_' + solution_parameter + '/gain_' + solution_parameter + str(1000 + i) + '.png')

        pyplot.close(fig1)

    execution_path = os.getcwd()
    print "Entering temporary directory"
    os.chdir(output_folder + '/temp_' + solution_parameter + '/')
    print "Generating video"
    subprocess.call(
        "ffmpeg -y -framerate 5 -start_number 1000 -i gain_" + solution_parameter + "%d.png gain_" + solution_parameter + ".mp4",
        shell=True)
    os.chdir("..")

    print "Returning to " + output_folder + " directory"
    subprocess.call('cp temp_' + solution_parameter + '/gain_' + solution_parameter + ".mp4 .", shell=True)
    print "Cleaning up plots"
    subprocess.call("rm -r temp_" + solution_parameter, shell=True)

    os.chdir(execution_path)
    return


def CRAMPS_cube_loader(output_folder, solution_type, solution_parameter):
    print "loading " + output_folder + "/" + solution_type + "_" + solution_parameter + "_solutions.h5"
    solution_cube = h5py.File(output_folder + "/" + solution_type + "_" + solution_parameter + "_solutions.h5", 'r')
    solution_data = solution_cube['data'][:]
    solution_quantity = solution_cube['parameters'][:]
    l = solution_cube['source_locations'][:]
    solution_cube.close()

    return solution_data, solution_quantity, l


def solution_histogram_plotter(fig1, number_visibilities, number_tiles, solution_data, solution_parameter,
                               solution_quantity, l_index, color="b"):
    antenna_index = 1
    # General plot formatting tools
    labelfontsize = 14
    amplitude_plotscale = 'log'
    phase_plotscale = 'linear'
    number_bins = 100
    # plot counter for antennas and visibilities
    antennaplot = 1
    visibilityplot = 1

    nrow = 1
    ncol = 1 + number_visibilities

    for j in range(number_visibilities + number_tiles):
        max_solutions = numpy.max(solution_data[j, :, :])
        min_solutions = numpy.min(solution_data[j, :, :])

        # plot the amplitude data
        if j == antenna_index:
            subplot = fig1.add_subplot(nrow, ncol, 1)
            subplot.set_ylim([1e-1, 1e3])
            subplot.set_ylabel(r'Frequency', fontsize=labelfontsize)
            subplot.hist(solution_data[j, l_index, :], histtype='stepfilled', edgecolor='none', alpha=0.4,
                         bins=number_bins, facecolor=color)
            if solution_parameter == "amp":
                subplot.set_xlabel(r'$\| g_{%d} \|$' % antenna_index, fontsize=labelfontsize)
                subplot.set_yscale(amplitude_plotscale)
                subplot.set_xlim([min_solutions, max_solutions])

            elif solution_parameter == "phase":
                visibility_index = solution_quantity[j]
                subplot = fig1.add_subplot(nrow, ncol, 1 + visibilityplot)
                subplot.set_xlabel(r'$g_{\phi%d}$' % visibility_index, fontsize=labelfontsize)
                subplot.set_yscale(phase_plotscale)
                subplot.set_xlim([-max_solutions, max_solutions])

        elif j >= number_tiles:
            subplot = fig1.add_subplot(nrow, ncol, 1 + visibilityplot)
            subplot.set_ylim([1e-1, 1e3])
            subplot.hist(solution_data[j, l_index, :], histtype='stepfilled', edgecolor='none', alpha=0.4,
                         bins=number_bins,
                         facecolor=color)

            if solution_parameter == "amp":
                subplot.set_xlabel(r'$\| v_{%d} \|$' % antennaplot, fontsize=labelfontsize)
                subplot.set_yscale(amplitude_plotscale)
                subplot.set_xlim([min_solutions, max_solutions])

            elif solution_parameter == "phase":
                subplot.set_xlabel(r'$v_{\phi%d}$' % antennaplot, fontsize=labelfontsize)
                subplot.set_yscale(phase_plotscale)
                subplot.set_xlim([-max_solutions, max_solutions])

            visibilityplot += 1

    return fig1


def solution_averager(output_folder, save_to_disk, solution_type, solution_parameter):
    solution_data, solution_quantity, l = CRAMPS_cube_loader(output_folder, solution_type,
                                                                         solution_parameter)

    number_iterations = solution_data.shape[2]
    print number_iterations
    # Create empty tables, to save the results for each sky step
    data_means = table_setup(len(solution_quantity) + 1, len(l) + 1)
    data_devs = table_setup(len(solution_quantity) + 1, len(l) + 1)

    # Set the sky steps
    data_means[0, 1:] = l
    data_devs[0, 1:] = l

    # Set the antenna numbers
    data_means[1:, 0] = solution_quantity
    data_devs[1:, 0] = solution_quantity

    if number_iterations > 1:
        # calculate averages and standard deviations
        if save_to_disk[1] == 'median':
            data_means[1:, 1:] = numpy.median(solution_data, axis=2)
        else:
            sys.exit("save_to_disk[2] parameter should be 'median'")

        if save_to_disk[2] == 'std':
            data_devs[1:, 1:] = numpy.std(solution_data, axis=2)
        elif save_to_disk[2] == 'iqr':
            data_devs[1:, 1:] = numpy.subtract(*numpy.percentile(solution_data, [75, 25], axis=2))
        else:
            sys.exit("save_to_disk[3] parameter should be 'std' or 'iqr'")

    else:
        data_means[1:, 1:] = solution_data[:, :, 0]
        data_devs[1:, 1:] = 0

    numpy.savetxt(output_folder + "/" + solution_type + "_" + solution_parameter + "_" + save_to_disk[1] + ".txt",
                  data_means)
    numpy.savetxt(output_folder + "/" + solution_type + "_" + solution_parameter + "_" + save_to_disk[2] + ".txt",
                  data_devs)
    return

#
# def solution_averager(amp_solutions, phase_solutions, red_tiles, \
#                       red_groups, sky_coords, save_to_disk, direction, noise_param):
#     iterations = len(amp_solutions[0, 0, :])
#     # Create empty tables, to save the results for each sky step
#     amp_means = table_setup(len(red_tiles) + len(red_groups) + 1, len(sky_coords) + 1)
#     amp_devs = table_setup(len(red_tiles) + len(red_groups) + 1, len(sky_coords) + 1)
#
#     phase_means = table_setup(len(red_tiles) + len(red_groups) + 1, len(sky_coords) + 1)
#     phase_devs = table_setup(len(red_tiles) + len(red_groups) + 1, len(sky_coords) + 1)
#     # Set the sky steps
#     amp_means[0, 1:] = sky_coords
#     amp_devs[0, 1:] = sky_coords
#     phase_means[0, 1:] = sky_coords
#     phase_devs[0, 1:] = sky_coords
#
#     # Set the antenna numbers
#     amp_means[1:, 0] = numpy.concatenate((red_tiles, red_groups))
#     amp_devs[1:, 0] = numpy.concatenate((red_tiles, red_groups))
#     phase_means[1:, 0] = numpy.concatenate((red_tiles, red_groups))
#     phase_devs[1:, 0] = numpy.concatenate((red_tiles, red_groups))
#
#     if iterations > 1:
#         # calculate averages and standard deviations
#
#         if save_to_disk[2] == 'median':
#             amp_means[1:, 1:] = numpy.median(amp_solutions, axis=2)
#             phase_means[1:, 1:] = numpy.median(phase_solutions, axis=2)
#         else:
#             sys.exit("save_to_disk[2] parameter should be 'median'")
#
#         if save_to_disk[3] == 'std':
#             amp_devs[1:, 1:] = numpy.std(amp_solutions, axis=2)
#             phase_devs[1:, 1:] = numpy.std(phase_solutions, axis=2)
#         elif save_to_disk[3] == 'iqr':
#             amp_devs[1:, 1:] = numpy.subtract(*numpy.percentile(amp_solutions, [75, 25], axis=2))
#             phase_devs[1:, 1:] = numpy.subtract(*numpy.percentile(phase_solutions, [75, 25], axis=2))
#         else:
#             sys.exit("save_to_disk[3] parameter should be 'std' or 'iqr'")
#
#     else:
#         amp_means[1:, 1:] = amp_solutions[:, :, 0]
#         amp_devs[1:, 1:] = 0
#         phase_means[1:, 1:] = phase_solutions[:, :, 0]
#         phase_devs[1:, 1:] = 0
#
#     if save_to_disk[0]:
#         save_to_text(save_to_disk, [amp_means, amp_devs], \
#                      [phase_means, phase_devs], noise_param[0], direction=direction)
#     return [amp_means, amp_devs], [phase_means, phase_devs]
