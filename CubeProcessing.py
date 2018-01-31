import numpy
import h5py
import os
import subprocess
import sys
from matplotlib import pyplot
from matplotlib import rcParams
from matplotlib import rc

rcParams['mathtext.default'] = 'regular'
rcParams.update({'figure.autolayout': True})

"""
This module contains all functions required to process the full solutions hdf5 files
*solution histogram plotter makes videos of the cube as the sources moves along the sky
*Solution averager calculates the mean/median and std/iqr

"""


def cube_processor(output_path,simulation_run,simulation_type,histogram_plotset,solution_averaging):
    if simulation_type == "CRAMPS":
        if histogram_plotset:
            CRAMPS_histogram_inspection(output_path + simulation_run, histogram_plotset)
        else:
            sys.exit("blaah")
    elif simulation_type == "SiSpS":
        if histogram_plotset:
            SiSps_histogram_inspection(output_path + simulation_run,histogram_plotset)
        else:
            sys.exit("blaah")
    else:
        sys.exit("Simulation type unknown: Please choose 'CRAMPS' or 'SiSpS'")
    return

def SiSps_histogram_inspection(output_folder, solution_type):
    if solution_type == "ideal":
        create_solution_histogram_tile(output_folder, "ideal", "amp")
        create_solution_histogram_tile(output_folder, "ideal", "phase")
    elif solution_type == "noisy":
        create_solution_histogram_tile(output_folder, "noisy", "amp")
        create_solution_histogram_tile(output_folder, "noisy", "phase")
    elif solution_type == "both":
        create_solution_histogram_tile(output_folder, "both", "amp")
        create_solution_histogram_tile(output_folder, "both", "phase")
    else:
        sys.exit("solution_type: please select 'ideal','noisy' or 'both', Goodbye.")
    return

def create_solution_histogram_tile(output_folder, solution_type, solution_parameter):
    print ""
    print "Loading data"
    if solution_type == "ideal":
        ideal_solution_data, parameters, position_offsets, peak_fluxes, ideal_iterations = \
            SiSpS_cube_loader(output_folder,  "ideal", solution_parameter)

        #indices = numpy.where(abs(ideal_solution_quantity) > 5e7)[0]
        #number_visibilities = len(indices)
        #number_tiles = len(ideal_solution_quantity) - number_visibilities

    elif solution_type == "noisy":
        noisy_solution_data, parameters, position_offsets, peak_fluxes, noisy_iterations = \
            SiSpS_cube_loader(output_folder, "noisy", solution_parameter)

        #indices = numpy.where(abs(noisy_solution_quantity) > 5e7)[0]
        #number_visibilities = len(indices)
        #number_tiles = len(noisy_solution_quantity) - number_visibilities

    elif solution_type == "both":
        ideal_solution_data, ideal_solution_quantity, ideal_position_offsets, ideal_peak_fluxes, ideal_iterations = \
            SiSpS_cube_loader(output_folder,  "ideal", solution_parameter)
        noisy_solution_data, noisy_solution_quantity, noisy_position_offsets, noisy_peak_fluxes, noisy_iterations = \
            SiSpS_cube_loader(output_folder, "noisy", solution_parameter)
        ideal_indices = numpy.where(abs(ideal_solution_quantity) > 5e7)[0]
        noisy_indices = numpy.where(abs(ideal_solution_quantity) > 5e7)[0]
        if len(ideal_indices) != len(noisy_indices):
            sys.exit("Number of redundant visibilities differs in ideal and noisy files, something is wrong")
        else:
            parameters = ideal_solution_quantity
            position_offsets = ideal_position_offsets
            peak_fluxes = ideal_peak_fluxes
            number_visibilities = len(ideal_indices)
            #number_tiles = len(ideal_solution_quantity) - number_visibilities
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
            if quantity <5e7:
                print "Antenna: ", int(quantity)
            else:
                print "Visibility: ", int(quantity)
        print "To exit : q"
        user_choice = raw_input("Your choice of the day :")

        if user_choice == "q":
            break
        else:
            user_quantity = int(user_choice)
            quantity_index = numpy.where(user_quantity == parameters.astype(int))[0]

            #pass this along to plotting function which creates a tile
            print "Generating plots"
            fig1 = pyplot.figure(figsize=(4*len(position_offsets), 4*len(peak_fluxes)))
            if solution_type == "ideal":
                fig1 = plot_solution_histogram_tile(fig1, parameters[quantity_index], ideal_solution_data[quantity_index,:,:,:], position_offsets, peak_fluxes)
            elif solution_type == "noisy":
                fig1 = plot_solution_histogram_tile(fig1, parameters[quantity_index], noisy_solution_data[quantity_index,:,:,:], position_offsets, peak_fluxes)
            elif solution_type == "both":
                fig1 = plot_solution_histogram_tile(fig1, parameters[quantity_index], ideal_solution_data[quantity_index,:,:,:], position_offsets, peak_fluxes, "r")
                fig1 = plot_solution_histogram_tile(fig1, parameters[quantity_index], noisy_solution_data[quantity_index,:,:,:], position_offsets, peak_fluxes, "b")


            fig1.suptitle(r'' + str(parameters[quantity_index]) + 'solutions', y=1.001)
            pyplot.show()

    return


def plot_solution_histogram_tile(fig1, quantity_number, solution_data, position_offsets, peak_fluxes, color= "b"):
    # General plot formatting tools
    labelfontsize = 14
    amplitude_plotscale = 'log'
    phase_plotscale = 'linear'
    number_bins = 100
    stepsize = 2

    rows = numpy.arange(0,len(position_offsets),2)
    cols = numpy.arange(0,len(peak_fluxes),2)
    nrow = len(rows)
    ncol = len(cols)
    plotcounter = 1
    for offset_index in rows:
        for flux_index in cols:
            subplot = fig1.add_subplot(nrow, ncol, plotcounter)
            # print offset_index, flux_index
            # print solution_data.shape
            # print solution_data[:, :, :].shape
            # print solution_data[offset_index, :, :].shape
            #print solution_data[0,offset_index, flux_index, :].shape


            subplot.hist(solution_data[0, offset_index, flux_index, :], histtype='stepfilled', edgecolor='none', alpha=0.4,
                         bins=number_bins, facecolor=color)

            subplot.text(0.95, 0.01, r'$\sigma =%s$' % (str(numpy.log10(position_offsets[offset_index]))),
                         verticalalignment='bottom', horizontalalignment='right',
                         transform=subplot.transAxes, fontsize=15)
            subplot.text(0.95, 0.21, r'$S =  %s Jy$' % (str(numpy.log10(peak_fluxes[flux_index]))),
                         verticalalignment='bottom', horizontalalignment='right',
                         transform=subplot.transAxes, fontsize=15)
            minimum = numpy.min(solution_data[0, offset_index, flux_index, :])
            maximum = numpy.max(solution_data[0, offset_index, flux_index, :])

            subplot.set_xlim([minimum, maximum])
            plotcounter += 1

    return fig1



def SiSpS_cube_loader(output_folder,solution_type,solution_parameter):
    print output_folder + "/" + solution_type + "_" + solution_parameter + "_solutions.h5"

    solution_cube = h5py.File(output_folder + "/" + solution_type + "_" + solution_parameter + "_solutions.h5", 'r')
    solution_data = solution_cube['data'][:]
    solution_quantity = solution_cube['parameters'][:]
    position_offsets = solution_cube['positions_uncertainty'][:]
    peak_fluxes = solution_cube['peak_flux'][:]
    iterations = solution_cube['iteration'][:]
    solution_cube.close()

    return solution_data,solution_quantity,position_offsets, peak_fluxes, iterations


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
    if not os.path.exists(output_folder + '/temp_'+solution_parameter+'/'):
        print ""
        print "Creating temporary output folder in "+output_folder+" directory"
        os.makedirs(output_folder + '/temp_'+solution_parameter+'/')

    print ""
    print "Loading data"
    if solution_type == "ideal":
        ideal_solution_data, ideal_solution_quantity, l, ideal_iterations = CRAMPS_cube_loader(output_folder, solution_type, solution_parameter)
        indices = numpy.where(abs(ideal_solution_quantity) > 5e7)[0]
        number_visibilities = len(indices)
        number_tiles = len(ideal_solution_quantity) - number_visibilities

    elif solution_type == "noisy":
        noisy_solution_data, noisy_solution_quantity, l, noisy_iterations = CRAMPS_cube_loader(output_folder, solution_type, solution_parameter)
        indices = numpy.where(abs(noisy_solution_quantity) > 5e7)[0]
        number_visibilities = len(indices)
        number_tiles = len(noisy_solution_quantity) - number_visibilities

    elif solution_type == "both":
        ideal_solution_data, ideal_solution_quantity, ideal_l, ideal_iterations = CRAMPS_cube_loader(output_folder, "ideal", solution_parameter)
        noisy_solution_data, noisy_solution_quantity, noisy_l, noisy_iterations = CRAMPS_cube_loader(output_folder, "noisy", solution_parameter)
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
                                               solution_parameter, ideal_solution_quantity, i,"r")
            fig1 = solution_histogram_plotter(fig1, number_visibilities, number_tiles, noisy_solution_data,
                                              solution_parameter, noisy_solution_quantity, i,"b")

        fig1.suptitle(r'' + solution_parameter + 'solutions at $\mathit{l}$=' + str(l[i]), y=1.001)
        fig1.savefig(output_folder + '/temp_'+solution_parameter+'/gain_' + solution_parameter + str(1000 + i) + '.png')

        pyplot.close(fig1)

    execution_path = os.getcwd()
    print "Entering temporary directory"
    os.chdir(output_folder + '/temp_'+solution_parameter+'/')
    print "Generating video"
    subprocess.call(
        "ffmpeg -y -framerate 5 -start_number 1000 -i gain_" + solution_parameter + "%d.png gain_" + solution_parameter + ".mp4",
        shell=True)
    os.chdir("..")

    print "Returning to "+output_folder+" directory"
    subprocess.call('cp temp_'+solution_parameter+'/gain_' + solution_parameter + ".mp4 .", shell=True)
    print "Cleaning up plots"
    subprocess.call("rm -r temp_"+solution_parameter, shell=True)

    os.chdir(execution_path)
    return

def CRAMPS_cube_loader(output_folder,solution_type,solution_parameter):
    solution_cube = h5py.File(output_folder + "/" + solution_type + "_" + solution_parameter + "_solutions.h5", 'r')
    solution_data = solution_cube['data'][:]
    solution_quantity = solution_cube['parameters'][:]
    l = solution_cube['l_coordinates'][:]
    iterations = solution_cube['iteration'][:]
    solution_cube.close()

    return solution_data,solution_quantity,l,iterations


def solution_histogram_plotter(fig1, number_visibilities, number_tiles, solution_data, solution_parameter,
                               solution_quantity,l_index, color= "b"):
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
            subplot.hist(solution_data[j, l_index, :], histtype='stepfilled', edgecolor='none', alpha=0.4, bins=number_bins,
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
