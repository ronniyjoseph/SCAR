import numpy
import h5py
import os
import subprocess

"""
This module contains all functions required to process the full solutions hdf5 files
*solution histogram plotter makes videos of the cube as the sources moves along the sky
*Solution averager calculates the mean/median and std/iqr

"""

def solution_histogram_plotter(output_folder,solution_type):

    #General plot formatting tools
    labelfontsize = 14
    amplitude_plotscale = 'log'
    phase_plotscale = 'linear'


    if not os.path.exists(output_folder+"/temp"):
        print ""
        print "Creating temporary output folder in working directory"

        os.makedirs(output_folder)

    print ""
    print "loading data"


    solution_set = cube_loader(output_folder,solution_type)



    print ""
    print "Creating histogram video of the solutions"


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

        ampfig1.savefig(output_folder + '/temp/gain_amp_' + str(1000 + i) + '.png')
        phasefig1.savefig(output_folder + '/temp/gain_phase_' + str(1000 + i) + '.png')

        pyplot.close(ampfig1)
        pyplot.close(phasefig1)

    execution_path = os.getcwd()
    os.chdir(output_folder+'/temp/)
    print "Generating video"
    subprocess.call("ffmpeg -y -framerate 5 -start_number 1000 -i gain_amp_%d.png gain_amp.mp4", shell=True)
    subprocess.call("ffmpeg -y -framerate 5 -start_number 1000 -i gain_phase_%d.png gain_phase.mp4", shell=True)
    os.chdir(output_folder)
    subprocess.call("cp temp/gain_amp.mp4" ., shell=True)
    subprocess.call("cp temp/gain_phase_amp.mp4"., shell=True)
    print "Cleaning up plots"
    subprocess.call("rm -r temp", shell=True)
    subprocess.call("rm -r temp", shell=True)

    os.chdir(execution_path)


def cube_loader(output_folder,solution_type):
    if solution_type == "ideal":
        amp_solutions = h5py.File(output_folder+files[plotnumber],'r')

    elif solution_type == "noisy":

    elif solution_type == "both":

    else:
        sys.exit("solution_type: please select 'ideal','noisy' or 'both', Goodbye.")

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