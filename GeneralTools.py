import numpy
import h5py
import os
import subprocess
from scipy import interpolate
import sys
import matplotlib
from matplotlib import pyplot
from matplotlib import rcParams
from matplotlib import rc

rcParams['mathtext.default'] = 'regular'
rcParams.update({'figure.autolayout': True})


def unique_value_finder(antenna_names, quantity):
    antennas = numpy.array(antenna_names)
    unique_antennas = numpy.unique(antennas)
    if quantity == 'number':
        output = len(unique_antennas)
    elif quantity == 'values':
        output = unique_antennas
    elif quantity == 'both':
        output = [unique_antennas, len(unique_antennas)]
    else:
        sys.exit("You should call 'unique value finder' with: number, values or both")
    return output


def position_finder(red_tiles, xy_positions):
    x_coordinates = numpy.zeros(len(red_tiles))
    y_coordinates = numpy.zeros(len(red_tiles))

    for i in range(len(red_tiles)):
        tile_index = numpy.where(xy_positions[:, 0] == red_tiles[i])

        x_coordinates[i] = xy_positions[tile_index[0], 1]
        y_coordinates[i] = xy_positions[tile_index[0], 2]
    return x_coordinates, y_coordinates


def solution_averager(amp_solutions, phase_solutions, red_tiles, \
                      red_groups, sky_coords, save_to_disk, direction, noise_param):
    iterations = len(amp_solutions[0, 0, :])
    # Create empty tables, to save the results for each sky step
    amp_means = table_setup(len(red_tiles) + len(red_groups) + 1, len(sky_coords) + 1)
    amp_devs = table_setup(len(red_tiles) + len(red_groups) + 1, len(sky_coords) + 1)

    phase_means = table_setup(len(red_tiles) + len(red_groups) + 1, len(sky_coords) + 1)
    phase_devs = table_setup(len(red_tiles) + len(red_groups) + 1, len(sky_coords) + 1)
    # Set the sky steps
    amp_means[0, 1:] = sky_coords
    amp_devs[0, 1:] = sky_coords
    phase_means[0, 1:] = sky_coords
    phase_devs[0, 1:] = sky_coords

    # Set the antenna numbers
    amp_means[1:, 0] = numpy.concatenate((red_tiles, red_groups))
    amp_devs[1:, 0] = numpy.concatenate((red_tiles, red_groups))
    phase_means[1:, 0] = numpy.concatenate((red_tiles, red_groups))
    phase_devs[1:, 0] = numpy.concatenate((red_tiles, red_groups))

    if iterations > 1:
        # calculate averages and standard deviations

        if save_to_disk[2] == 'median':
            amp_means[1:, 1:] = numpy.median(amp_solutions, axis=2)
            phase_means[1:, 1:] = numpy.median(phase_solutions, axis=2)
        else:
            sys.exit("save_to_disk[2] parameter should be 'median'")

        if save_to_disk[3] == 'std':
            amp_devs[1:, 1:] = numpy.std(amp_solutions, axis=2)
            phase_devs[1:, 1:] = numpy.std(phase_solutions, axis=2)
        elif save_to_disk[3] == 'iqr':
            amp_devs[1:, 1:] = numpy.subtract(*numpy.percentile(amp_solutions, [75, 25], axis=2))
            phase_devs[1:, 1:] = numpy.subtract(*numpy.percentile(phase_solutions, [75, 25], axis=2))
        else:
            sys.exit("save_to_disk[3] parameter should be 'std' or 'iqr'")

    else:
        amp_means[1:, 1:] = amp_solutions[:, :, 0]
        amp_devs[1:, 1:] = 0
        phase_means[1:, 1:] = phase_solutions[:, :, 0]
        phase_devs[1:, 1:] = 0

    if save_to_disk[0]:
        save_to_text(save_to_disk, [amp_means, amp_devs], \
                     [phase_means, phase_devs], noise_param[0], direction=direction)
    return [amp_means, amp_devs], [phase_means, phase_devs]


def FourD_solution_averager(amp_solutions, phase_solutions, red_groups,
                            red_tiles, peak_fluxes, sigma_offsets, save_to_disk):
    iterations = len(amp_solutions[0, 0, 0, :])

    # Set the antenna numbers
    solution_index = numpy.concatenate((red_tiles, red_groups))

    # calculate averages and standard deviations
    amp_accuracy = numpy.median(amp_solutions - 1, axis=3)
    amp_precision = numpy.std(amp_solutions, axis=3)
    # amp_devs[1:,1:] = numpy.subtract( \
    #	*numpy.percentile(amp_solutions,[75,25],axis=2))
    phase_accuracy = numpy.median(phase_solutions, axis=3)
    phase_precision = numpy.std(phase_solutions, axis=3)
    # phase_devs[1:,1:] = numpy.subtract( \
    # *numpy.percentile(phase_solutions,[75,25],axis=2))

    if save_to_disk[0]:
        save_to_hdf5(save_to_disk[1], 'amp_means', amp_accuracy, solution_index, sigma_offsets, peak_fluxes)
        save_to_hdf5(save_to_disk[1], 'amp_devs', amp_precision, solution_index, sigma_offsets, peak_fluxes)
        save_to_hdf5(save_to_disk[1], 'phase_means', phase_accuracy, solution_index, sigma_offsets, peak_fluxes)
        save_to_hdf5(save_to_disk[1], 'phase_devs', phase_precision, solution_index, sigma_offsets, peak_fluxes)

    return [amp_accuracy, amp_precision], [phase_accuracy, phase_precision]


def table_setup(size_x, size_y):
    table = numpy.zeros((size_x, size_y))
    table[0, 0] = numpy.nan
    return table


def position_finder(red_tiles, xy_positions):
    x_coordinates = numpy.zeros(len(red_tiles))
    y_coordinates = numpy.zeros(len(red_tiles))

    for i in range(len(red_tiles)):
        tile_index = numpy.where(xy_positions[:, 0] == red_tiles[i])

        x_coordinates[i] = xy_positions[tile_index[0], 1]
        y_coordinates[i] = xy_positions[tile_index[0], 2]
    return x_coordinates, y_coordinates


def save_to_hdf5(pathfolder, fname, savedata, axesdata, axeslabels):
    """
	:type pathfolder: string containing the path of output destination
	"""
    # Check whether output folder is present
    if not os.path.exists(pathfolder):
        print ""
        print "!!!Warning: Creating output folder in working directory!!!"
        os.makedirs(pathfolder)

    if len(axesdata) != len(axeslabels):
        sys.exit("Data Axes length (" + str(len(axesdata)) + ") does not equal Axes Label length (" + str(
            len(axeslabels)) + ")")

    datafile = h5py.File(pathfolder + fname + '.h5', 'w')
    datafile['data'] = savedata
    for index in range(len(axesdata)):

        datafile[axeslabels[index]] = axesdata[index]
    datafile.close()
    return


def save_to_text(save_params, amp_data, phase_data, noisy, direction):
    # Check whether output folder is present
    version = save_params[1]

    if not os.path.exists(version):
        print ""
        print "!!!Warning: Creating output folder in working directory!!!"
        os.makedirs(version)

    if noisy:
        numpy.savetxt(version + '/noisy_' + direction + '_amp_means.txt', amp_data[0])
        numpy.savetxt(version + '/noisy_' + direction + '_amp_devs.txt', amp_data[1])

        numpy.savetxt(version + '/noisy_' + direction + '_phase_means.txt', phase_data[0])
        numpy.savetxt(version + '/noisy_' + direction + '_phase_devs.txt', phase_data[1])
    else:
        numpy.savetxt(version + '/ideal_' + direction + '_amp_means.txt', amp_data[0])
        numpy.savetxt(version + '/ideal_' + direction + '_amp_devs.txt', amp_data[1])

        numpy.savetxt(version + '/ideal_' + direction + '_phase_means.txt', phase_data[0])
        numpy.savetxt(version + '/ideal_' + direction + '_phase_devs.txt', phase_data[1])
    return


def visibility_histogram_plotter(amp_obs, phase_obs, amp_mod, phase_mod, \
                                 sky_steps, noisy_amp_info_l, noisy_phase_info_l, output_folder):
    if not os.path.exists(output_folder):
        print ""
        print "!!!Warning: Creating output folder in working directory!!!"
        os.makedirs(output_folder)

    print ""
    print "Creating histogram video of the visibilities"

    labelfontsize = 14

    number_visibilities = len(amp_obs[:, 1, 1])
    l = sky_steps

    print "Preprocessing the data"
    min_amp_obs = numpy.zeros((number_visibilities, len(sky_steps)))
    min_amp_mod = numpy.zeros((number_visibilities, len(sky_steps)))

    mean_amp_obs = numpy.zeros((number_visibilities, len(sky_steps)))
    mean_amp_mod = numpy.zeros((number_visibilities, len(sky_steps)))

    max_amp_obs = numpy.zeros((number_visibilities, len(sky_steps)))
    max_amp_mod = numpy.zeros((number_visibilities, len(sky_steps)))
    # Pre-processing loop to fit nice plot ranges
    for i in range(len(l)):
        min_amp_obs[:, i] = numpy.amin(amp_obs[:, i, :], axis=1)
        min_amp_mod[:, i] = numpy.amin(amp_mod[:, i, :], axis=1)

        mean_amp_obs[:, i] = numpy.mean(amp_obs[:, i, :], axis=1)
        mean_amp_mod[:, i] = numpy.mean(amp_mod[:, i, :], axis=1)

        max_amp_obs[:, i] = numpy.amax(amp_obs[:, i, :], axis=1)
        max_amp_mod[:, i] = numpy.amax(amp_mod[:, i, :], axis=1)

    # Fit the the actual smoothed plotranges
    max_smooth = numpy.zeros((number_visibilities, len(sky_steps)))
    min_smooth = numpy.zeros((number_visibilities, len(sky_steps)))
    for j in range(number_visibilities):
        max_spline = interpolate.UnivariateSpline(l, mean_amp_obs[j, :] + \
                                                  1.5 * abs(mean_amp_obs[j, :] - max_amp_obs[0, :]), k=3, s=10000000)
        min_spline = interpolate.UnivariateSpline(l, mean_amp_obs[j, :] - \
                                                  1.5 * abs(mean_amp_obs[j, :] - max_amp_obs[0, :]), k=3, s=10000000)

        max_smooth[j, :] = max_spline(l)
        min_smooth[j, :] = min_spline(l)

    # set all negative values to zero
    min_smooth[min_smooth < 0] = -10.

    # Create the actual okits
    print "Generating plots"
    for i in range(len(l)):
        ampfig1 = pyplot.figure(figsize=(16, 10))
        phasefig1 = pyplot.figure(figsize=(16, 10))

        nrow = 2
        ncol = 4

        # plot counter for antennas and visibilities
        visibilityplot = 1

        ampfig1.suptitle(r'Visibility Amplitudes at $\mathit{l}$=' + str(l[i]), y=1.001)
        phasefig1.suptitle(r'Visibility Phases at $\mathit{l}$=' + str(l[i]), y=1.001)
        for j in range(number_visibilities):

            ampsub = ampfig1.add_subplot(nrow, ncol, visibilityplot)
            phasesub = phasefig1.add_subplot(nrow, ncol, visibilityplot)

            ampsub.set_xlabel(r'$\| v_{%d} \|$' % visibilityplot, fontsize=labelfontsize)
            phasesub.set_xlabel(r'$v_{\phi%d}$' % visibilityplot, fontsize=labelfontsize)

            max_amp = max_smooth[j, i]
            min_amp = min_smooth[j, i]
            max_phase = numpy.max(phase_obs[j, :, :])

            ampsub.set_xlim([min_amp, max_amp])
            phasesub.set_xlim([-1.25 * max_phase, 1.25 * max_phase])

            ampsub.set_ylim([1e-1, 1e3])
            phasesub.set_ylim([1e-1, 1e3])

            ampsub.set_yscale('log')
            # ampsub.set_xscale('log')

            phasesub.set_yscale('log')

            if visibilityplot == 1 or visibilityplot == 5:
                ampsub.set_ylabel(r'Frequency', fontsize=labelfontsize)
                phasesub.set_ylabel(r'Frequency', fontsize=labelfontsize)

            visibilityplot += 1

            # Plot solutions histograms

            # ampbins = numpy.logspace(numpy.log10(min(amp_obs[j,i,:])),numpy.log10(max(amp_obs[j,i,:])),50)
            # phasebins = numpy.linspace(min(phase_obs[j,i,:]),max(phase_obs[j,i,:]),50)

            ampsub.hist(amp_obs[j, i, :], histtype='stepfilled', alpha=0.3, bins=50, facecolor='b')
            phasesub.hist(phase_obs[j, i, :], histtype='stepfilled', alpha=0.3, bins=50, facecolor='b')

            # ampbins = numpy.logspace(numpy.log10(min(amp_mod[j,i,:])),numpy.log10(max(amp_mod[j,i,:])),50)
            # phasebins = numpy.linspace(min(phase_mod[j,i,:]),max(phase_mod[j,i,:]),50)

            ampsub.hist(amp_mod[j, i, :], histtype='stepfilled', alpha=0.3, bins=50, facecolor='r')
            phasesub.hist(phase_mod[j, i, :], histtype='stepfilled', alpha=0.3, bins=50, facecolor='r')

        ################################################################
        ################################################################
        ####################### Plot with solutions ####################

        ampsub = ampfig1.add_subplot(nrow, ncol, nrow * ncol)
        phasesub = phasefig1.add_subplot(nrow, ncol, nrow * ncol)
        tile_number = 2

        ################################################################
        ###############				Amplitude			################
        ################################################################
        x = noisy_amp_info_l[0][0, 1:]
        y = noisy_amp_info_l[0][tile_number, 1:]
        y_plus = noisy_amp_info_l[0][tile_number, 1:] + noisy_amp_info_l[1][tile_number, 1:]
        y_min = noisy_amp_info_l[0][tile_number, 1:] - noisy_amp_info_l[1][tile_number, 1:]

        ampsub.plot(x, y, color="SteelBlue")
        ampsub.plot(l[i], 1, "ko")
        ampsub.axhline(y=1, xmin=min(x), xmax=max(x), color='k')

        ampsub.fill_between(x, y_plus, y_min, color="SteelBlue", alpha=0.3)
        ampsub.set_xlabel(r'$\mathit{l}$', fontsize=labelfontsize)
        ampsub.set_ylabel(r'$\| g_{' + str(tile_number) + '} \|$', fontsize=labelfontsize)

        amprange = numpy.round(max([abs(min(y_min)), abs(max(y_plus))]), 1)
        amprange += 0.1 * amprange

        ampsub.set_ylim([0, amprange])

        ################################################################
        ###############				Phase				################
        ################################################################
        # Extract the values for the x-axis: sky coordinates
        x = noisy_phase_info_l[0][0, 1:]
        # Extract the values the the y axis: solution mean
        y = noisy_phase_info_l[0][1, 1:]
        # Create shaded 1-sigma area
        y_plus = noisy_phase_info_l[0][tile_number, 1:] + noisy_phase_info_l[1][tile_number, 1:]
        y_min = noisy_phase_info_l[0][tile_number, 1:] - noisy_phase_info_l[1][tile_number, 1:]

        phasesub.plot(x, y, color="SteelBlue")
        phasesub.plot(l[i], 0, "ko")
        phasesub.fill_between(x, y_plus, y_min, color="SteelBlue", alpha=0.3)
        phasesub.axhline(y=0, xmin=min(x), xmax=max(x), color='k')

        phasesub.set_xlabel(r'$\mathit{l}$', fontsize=labelfontsize)
        phasesub.set_ylabel(r'$\phi_{' + str(tile_number) + '}\, [rad]$', fontsize=labelfontsize)

        phaserange = numpy.round(max([abs(min(y_min)), abs(max(y_plus))]), 1)
        phaserange += 0.5 * phaserange
        phasesub.set_ylim([-phaserange, phaserange])

        ampfig1.savefig(output_folder + 'vis_amp_' + str(1000 + i) + '.png')
        phasefig1.savefig(output_folder + 'vis_phase_' + str(1000 + i) + '.png')

        pyplot.close(ampfig1)
        pyplot.close(phasefig1)

    execution_path = os.getcwd()
    os.chdir(output_folder)
    print "Generating video"
    subprocess.call("ffmpeg -y -framerate 5 -start_number 1000 -i vis_amp_%d.png vis_amp.mp4", shell=True)
    subprocess.call("ffmpeg -y -framerate 5 -start_number 1000 -i vis_phase_%d.png vis_phase.mp4", shell=True)
    print "Cleaning up plots"
    subprocess.call("rm vis_phase_*.png", shell=True)
    subprocess.call("rm vis_amp_*.png", shell=True)

    os.chdir(execution_path)


def solution_histogram_plotter(amp_solutions, phase_solutions, noisy_amp_info_l, noisy_phase_info_l, output_folder):
    if not os.path.exists(output_folder):
        print ""
        print "!!!Warning: Creating output folder in working directory!!!"
        os.makedirs(output_folder)

    print ""
    print "Creating histogram video of the solutions"

    labelfontsize = 14

    indices = numpy.where(abs(noisy_amp_info_l[0][:, 0]) > 5e7)[0]

    number_visibilities = len(indices)
    number_tiles = len(noisy_amp_info_l[0][:, 0]) - 1 - len(indices)
    l = noisy_amp_info_l[0][0, 1:]
    red_tiles = noisy_amp_info_l[0][1:, 0]

    # max_gain_amp = numpy.max(amp_solutions[0:number_tiles,:,:])
    # max_vis_amp  = numpy.max(amp_solutions[number_tiles:number_tiles+number_visibilities,:,:])

    # max_gain_phase = numpy.max(phase_solutions[0:number_tiles,:,:])
    # max_vis_phase  = numpy.max(phase_solutions[number_tiles:number_tiles+number_visibilities,:,:])

    print "Generating plots"
    for i in range(len(l)):
        ampfig1 = pyplot.figure(figsize=(16, 10))
        phasefig1 = pyplot.figure(figsize=(16, 10))

        nrow = 2
        ncol = number_tiles

        # plot counter for antennas and visibilities
        antennaplot = 1
        visibilityplot = 1

        ampfig1.suptitle(r'Amplitude solutions at $\mathit{l}$=' + str(l[i]), y=1.001)
        phasefig1.suptitle(r'Phase solutions at $\mathit{l}$=' + str(l[i]), y=1.001)
        for j in range(number_visibilities + number_tiles):

            # plot the amplitude data
            if j < number_tiles:
                ampsub = ampfig1.add_subplot(nrow, ncol, antennaplot)
                phasesub = phasefig1.add_subplot(nrow, ncol, antennaplot)
                # ~ ampsub.set_title("Antenna "+str(int(red_tiles[j]-1000)))
                # ~ phasesub.set_title("Antenna "+str(int(red_tiles[j]-1000)))

                ampsub.set_xlabel(r'$\| g_{%d} \|$' % antennaplot, fontsize=labelfontsize)
                phasesub.set_xlabel(r'$g_{\phi%d}$' % antennaplot, fontsize=labelfontsize)

                max_amp = numpy.max(amp_solutions[j, :, :])
                min_amp = numpy.min(amp_solutions[j, :, :])
                max_phase = numpy.max(phase_solutions[j, :, :])
                ampsub.set_yscale('log')

                ampsub.set_xlim([min_amp, max_amp])
                phasesub.set_xlim([-max_phase, max_phase])

                ampsub.set_ylim([1e-1, 1e3])
                phasesub.set_ylim([1e-1, 1e3])

                ampsub.set_yscale('log')
                phasesub.set_yscale('log')

                if antennaplot == 1:
                    ampsub.set_ylabel(r'Frequency', fontsize=labelfontsize)
                    phasesub.set_ylabel(r'Frequency', fontsize=labelfontsize)
                antennaplot += 1
            else:
                ampsub = ampfig1.add_subplot(nrow, ncol, ncol + visibilityplot)
                phasesub = phasefig1.add_subplot(nrow, ncol, ncol + visibilityplot)

                ampsub.set_xlabel(r'$\| v_{%d} \|$' % visibilityplot, fontsize=labelfontsize)
                phasesub.set_xlabel(r'$v_{\phi%d}$' % visibilityplot, fontsize=labelfontsize)

                max_amp = numpy.max(amp_solutions[j, :, :])
                min_amp = numpy.min(amp_solutions[j, :, :])
                max_phase = numpy.max(phase_solutions[j, :, :])
                ampsub.set_yscale('log')

                ampsub.set_xlim([min_amp, max_amp])
                phasesub.set_xlim([-max_phase, max_phase])

                ampsub.set_ylim([1e-1, 1e3])
                phasesub.set_ylim([1e-1, 1e3])

                ampsub.set_yscale('log')
                phasesub.set_yscale('log')

                if visibilityplot == 1:
                    ampsub.set_ylabel(r'Frequency', fontsize=labelfontsize)
                    phasesub.set_ylabel(r'Frequency', fontsize=labelfontsize)

                visibilityplot += 1

            # ~ index = indices[counter]

            # Plot solutions histograms
            ampsub.hist(amp_solutions[j, i, :], histtype='stepfilled', edgecolor='none', alpha=0.7, bins=500,
                        facecolor='blue')
            phasesub.hist(phase_solutions[j, i, :], histtype='stepfilled', edgecolor='none', alpha=0.7, bins=500,
                          facecolor='blue')

        ################################################################
        ################################################################
        ####################### Plot with solutions ####################

        ampsub = ampfig1.add_subplot(nrow, ncol, nrow * number_tiles)
        phasesub = phasefig1.add_subplot(nrow, ncol, nrow * number_tiles)
        tile_number = 2

        ################################################################
        ###############				Amplitude			################
        ################################################################
        x = noisy_amp_info_l[0][0, 1:]
        y = noisy_amp_info_l[0][tile_number, 1:]
        y_plus = noisy_amp_info_l[0][tile_number, 1:] + noisy_amp_info_l[1][tile_number, 1:]
        y_min = noisy_amp_info_l[0][tile_number, 1:] - noisy_amp_info_l[1][tile_number, 1:]

        ampsub.plot(x, y, color="SteelBlue")
        ampsub.plot(l[i], 1, "ko")
        ampsub.axhline(y=1, xmin=min(x), xmax=max(x), color='k')

        ampsub.fill_between(x, y_plus, y_min, color="SteelBlue", alpha=0.3)
        ampsub.set_xlabel(r'$\mathit{l}$', fontsize=labelfontsize)
        ampsub.set_ylabel(r'$\| g_{' + str(tile_number) + '} \|$', fontsize=labelfontsize)

        amprange = numpy.round(max([abs(min(y_min)), abs(max(y_plus))]), 1)
        amprange += 0.5 * amprange

        ampsub.set_ylim([0, amprange])

        ################################################################
        ###############				Phase				################
        ################################################################
        # Extract the values for the x-axis: sky coordinates
        x = noisy_phase_info_l[0][0, 1:]
        # Extract the values the the y axis: solution mean
        y = noisy_phase_info_l[0][1, 1:]
        # Create shaded 1-sigma area
        y_plus = noisy_phase_info_l[0][tile_number, 1:] + noisy_phase_info_l[1][tile_number, 1:]
        y_min = noisy_phase_info_l[0][tile_number, 1:] - noisy_phase_info_l[1][tile_number, 1:]

        phasesub.plot(x, y, color="SteelBlue")
        phasesub.plot(l[i], 0, "ko")
        phasesub.fill_between(x, y_plus, y_min, color="SteelBlue", alpha=0.3)
        phasesub.axhline(y=0, xmin=min(x), xmax=max(x), color='k')

        phasesub.set_xlabel(r'$\mathit{l}$', fontsize=labelfontsize)
        phasesub.set_ylabel(r'$\phi_{' + str(tile_number) + '}\, [rad]$', fontsize=labelfontsize)

        phaserange = numpy.round(max([abs(min(y_min)), abs(max(y_plus))]), 1)
        phaserange += 0.5 * phaserange
        phasesub.set_ylim([-phaserange, phaserange])

        ampfig1.savefig(output_folder + 'gain_amp_' + str(1000 + i) + '.png')
        phasefig1.savefig(output_folder + 'gain_phase_' + str(1000 + i) + '.png')

        pyplot.close(ampfig1)
        pyplot.close(phasefig1)

    execution_path = os.getcwd()
    os.chdir(output_folder)
    print "Generating video"
    subprocess.call("ffmpeg -y -framerate 5 -start_number 1000 -i gain_amp_%d.png gain_amp.mp4", shell=True)
    subprocess.call("ffmpeg -y -framerate 5 -start_number 1000 -i gain_phase_%d.png gain_phase.mp4", shell=True)
    print "Cleaning up plots"
    subprocess.call("rm gain_phase_*.png", shell=True)
    subprocess.call("rm gain_amp_*.png", shell=True)

    os.chdir(execution_path)


def TrueSolutions_Organizer(gain_table, model_vis, red_baseline_table, red_tiles, red_groups):
    red_amp_gains = numpy.zeros(len(red_tiles))
    red_phase_gains = numpy.zeros(len(red_tiles))

    red_visibilities = numpy.zeros(len(red_groups), 'complex')

    for i in range(len(red_groups)):
        indices = numpy.where(red_baseline_table[:, 7] == red_groups[i])[0]
        red_visibilities[i] = model_vis[indices[0], 0]

    for i in range(len(red_tiles)):
        indices = numpy.where(gain_table[:, 0, 0] == red_tiles[i])[0]
        red_amp_gains[i] = gain_table[indices[0], 1, 0]
        red_phase_gains[i] = gain_table[indices[0], 2, 0]

    amp_solutions = numpy.hstack((red_amp_gains, numpy.abs(red_visibilities)))
    phase_solutions = numpy.hstack((red_phase_gains, numpy.angle(red_visibilities)))

    true_solutions = amp_solutions * numpy.exp(1j * phase_solutions)
    return true_solutions
