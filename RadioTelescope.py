import numpy
import scipy.constants


def xyz_position_creator(shape):
    """
	Generates an array lay-out defined by input parameters, returns
	x,y,z coordinates of each antenna in the array
	
	shape	: list of array parameters
	shape[0]	: string value 'square', 'hex', 'doublehex', 'linear'
	
		'square': produces a square array
			shape[1]: 1/2 side of the square in meters
			shape[2]: minimum baseline length
			shape[3]: x position of square
			shape[4]: y position of square
		
		'hex': produces a hex array
		
		'doublehex': produces a double hex array
		
		'linear': produces a linear array
			shape[1]: x-outeredges of the array
			shape[2]: number of elements in the EW-linear array

	"""
    if shape[0] == "square" or shape[0] == 'doublesquare':
        print ""
        print "Creating x- y- z-positions of a square array"
        x_coordinates = numpy.arange(-shape[1], shape[1], shape[2])
        y_coordinates = numpy.arange(-shape[1], shape[1], shape[2])

        block1 = numpy.zeros((len(x_coordinates) * len(y_coordinates), 4))
        k = 0
        for i in range(len(x_coordinates)):
            for j in range(len(y_coordinates)):
                block1[k, 0] = 1001 + k
                block1[k, 1] = x_coordinates[i]
                block1[k, 2] = y_coordinates[j]
                block1[k, 3] = 0
                k += 1
        if shape[0] == 'square':
            block1[:, 1] += shape[3]
            block1[:, 2] += shape[4]
            xyz_coordinates = block1.copy()
        elif shape[0] == 'doublesquare':
            block2 = block1.copy()

            block2[:, 0] += 1000 + len(block1[:, 0])
            block2[:, 1] += shape[3]
            block2[:, 2] += shape[4]
            xyz_coordinates = numpy.vstack((block1, block2))

    elif shape[0] == 'hex' or shape[0] == 'doublehex':
        print ""
        print "Creating x- y- z-positions of a " + shape[0] + " array"

        dx = shape[1]
        dy = dx * numpy.sqrt(3.) / 2.

        line1 = numpy.array([numpy.arange(4) * dx, numpy.zeros(4), numpy.zeros(4)]).transpose()

        # define the second line
        line2 = line1[0:3, :].copy()
        line2[:, 0] += dx / 2.
        line2[:, 1] += dy
        # define the third line
        line3 = line1[0:3].copy()
        line3[:, 1] += 2 * dy
        # define the fourth line
        line4 = line2[0:2, :].copy()
        line4[:, 1] += 2 * dy

        block1 = numpy.vstack((line1[1:], line2, line3, line4))

        block2 = numpy.vstack((line1[1:], line2, line3[1:], line4))
        block2[:, 0] *= -1

        block3 = numpy.vstack((line2, line3, line4))
        block3[:, 1] *= -1

        block4 = numpy.vstack((line2, line3[1:], line4))
        block4[:, 0] *= -1
        block4[:, 1] *= -1
        hex_block = numpy.vstack((block1, block2, block3, block4))

        if shape[0] == 'hex':
            hex_block[:, 0] += shape[2]
            hex_block[:, 1] += shape[3]
            antenna_numbers = numpy.arange(len(hex_block[:, 0])) + 1001
            xyz_coordinates = numpy.vstack((antenna_numbers, hex_block.T)).T
        elif shape[0] == 'doublehex':
            antenna_numbers = numpy.arange(len(hex_block[:, 0])) + 1001
            first_hex = numpy.vstack((antenna_numbers, hex_block.T)).T

            second_hex = first_hex.copy()

            first_hex[:, 1] += shape[2]
            first_hex[:, 2] += shape[3]

            second_hex[:, 0] += 1000 + len(first_hex[:, 0])
            second_hex[:, 1] += shape[4]
            second_hex[:, 2] += shape[5]
            xyz_coordinates = numpy.vstack((first_hex, second_hex))

    elif shape[0] == 'linear':
        print ""
        print "Creating x- y- z-positions of a " + str(shape[2]) + " element linear array"
        xyz_coordinates = numpy.zeros((shape[2], 4))
        xyz_coordinates[:, 0] = numpy.arange(shape[2]) + 1001
        xyz_coordinates[:, 1] = numpy.linspace(-shape[1], shape[1], shape[2])
    elif shape[0] == 'file':
        xyz_positions = antenna_table_loader(shape[1])

    return xyz_coordinates


def baseline_converter(xy_positions, gain_table, frequency_channels,verbose=True):
    if verbose:
        print ""
        print "Converting xyz to uvw-coordinates"

    # calculate the wavelengths of the adjecent channels
    wavelength_range = scipy.constants.c / frequency_channels
    # Count the number of antenna
    number_of_antenna = len(xy_positions[:, 0])
    # Calculate the number of possible baselines
    number_of_baselines = int(0.5 * number_of_antenna * (number_of_antenna - 1.))
    # count the number of channels
    n_channels = len(frequency_channels)
    # Create an empty array for the baselines
    # baselines x Antenna1, Antenna2, u, v, w, gain product, phase sum x channels
    uv_positions = numpy.zeros((number_of_baselines, 7, n_channels))

    if verbose:
        print ""
        print "Number of antenna =", number_of_antenna
        print "Total number of baselines =", number_of_baselines

    # arbitrary counter to keep track of the baseline table
    k = 0
    for i in range(number_of_antenna):
        for j in range(i + 1, number_of_antenna):
            # save the antenna numbers in the uv table
            uv_positions[k, 0, :] = xy_positions[i, 0]
            uv_positions[k, 1, :] = xy_positions[j, 0]

            # rescale and write uvw to multifrequency baseline table
            uv_positions[k, 2, :] = (xy_positions[i, 1] - xy_positions[j, 1]) / \
                                    wavelength_range
            uv_positions[k, 3, :] = (xy_positions[i, 2] - xy_positions[j, 2]) / \
                                    wavelength_range
            uv_positions[k, 4, :] = (xy_positions[i, 3] - xy_positions[j, 3]) / \
                                    wavelength_range

            # Find the gains
            amp_gain1 = gain_table[gain_table[:, 0, 0] == xy_positions[i, 0], 1, :][0]
            amp_gain2 = gain_table[gain_table[:, 0, 0] == xy_positions[j, 0], 1, :][0]

            phase_gain1 = gain_table[gain_table[:, 0, 0] == xy_positions[i, 0], 2, :][0]
            phase_gain2 = gain_table[gain_table[:, 0, 0] == xy_positions[j, 0], 2, :][0]
            # calculate the complex baseline gain
            uv_positions[k, 5, :] = amp_gain1 * amp_gain2
            uv_positions[k, 6, :] = -(phase_gain1 - phase_gain2)

            k += 1

    return uv_positions


def redundant_baseline_finder(uv_positions, baseline_direction,verbose=False):
    """
	"""

    ################################################################
    minimum_baselines = 3.
    wave_fraction = 1. / 6
    ################################################################

    n_baselines = uv_positions.shape[0]
    n_frequencies = uv_positions.shape[2]
    middle_index = (n_frequencies + 1) / 2 - 1
    # create empty table
    baseline_selection = numpy.zeros((n_baselines, 8, n_frequencies))
    # arbitrary counters
    # Let's find all the redundant baselines within our threshold
    group_counter = 0
    k = 0
    # Go through all antennas, take each antenna out and all antennas
    # which are part of the not redundant enough group
    while uv_positions.shape[0] > 0:
        # calculate uv separation at the calibration wavelength
        separation = numpy.sqrt(
            (uv_positions[:, 2, middle_index] - uv_positions[0, 2, middle_index]) ** 2. +
            (uv_positions[:, 3, middle_index] - uv_positions[0, 3, middle_index]) ** 2.)
        # find all baselines within the lambda fraction
        select_indices = numpy.where(separation <= wave_fraction)

        # is this number larger than the minimum number
        if len(select_indices[0]) >= minimum_baselines:
            # go through the selected baselines

            for i in range(len(select_indices[0])):
                # add antenna number
                baseline_selection[k, 0, :] = uv_positions[select_indices[0][i], 0, :]
                baseline_selection[k, 1, :] = uv_positions[select_indices[0][i], 1, :]
                # add coordinates uvw
                baseline_selection[k, 2, :] = uv_positions[select_indices[0][i], 2, :]
                baseline_selection[k, 3, :] = uv_positions[select_indices[0][i], 3, :]
                baseline_selection[k, 4, :] = uv_positions[select_indices[0][i], 4, :]
                # add the gains
                baseline_selection[k, 5, :] = uv_positions[select_indices[0][i], 5, :]
                baseline_selection[k, 6, :] = uv_positions[select_indices[0][i], 6, :]
                # add baseline group identifier
                baseline_selection[k, 7, :] = 50000000 + 52 * (group_counter + 1)

                k += 1
            group_counter += 1
        # update the list, take out the used antennas
        all_indices = numpy.arange(len(uv_positions))
        unselected_indices = numpy.setdiff1d(all_indices, select_indices[0])

        uv_positions = uv_positions[unselected_indices]

    if verbose:
        print "There are", k, "redundant baselines in this array."
        print "There are", group_counter, "redundant groups in this array"

    # find the filled entries
    non_zero_indices = numpy.where(baseline_selection[:, 0, 0] != 0)
    # remove the empty entries
    baseline_selection = baseline_selection[non_zero_indices[0], :, :]
    # Sort on length
    baseline_lengths = numpy.sqrt(baseline_selection[:, 2, middle_index] ** 2 \
                                  + baseline_selection[:, 3, middle_index] ** 2)

    sorted_baselines = baseline_selection[numpy.argsort(baseline_lengths), :, :]

    sorted_baselines = baseline_selection[numpy.argsort(sorted_baselines[:, 7, middle_index]), :, :]
    # sorted_baselines = sorted_baselines[numpy.argsort(sorted_baselines[:,1,middle_index]),:,:]
    # if we want only the EW select all the  uv positions around v = 0
    if baseline_direction == "EW":
        ew_indices = numpy.where(abs(sorted_baselines[:, 3, middle_index]) < 5. / wavelength)
        selected_baselines = sorted_baselines[ew_indices[0], :, :]
    elif baseline_direction == "NS":
        ns_indices = numpy.where(abs(sorted_baselines[:, 2, middle_index]) < 5. / wavelength)
        selected_baselines = sorted_baselines[ns_indices[0], :, :]
    elif baseline_direction == "ALL":
        selected_baselines = sorted_baselines
    else:
        sys.exit("The given redundant baseline direction is invalid:" + \
                 " please use 'EW', 'ALL'")
    return sorted_baselines


def antenna_gain_creator(xyz_positions, frequency_channels):
    """
	"""
    n_channels = len(frequency_channels)

    gain_table = numpy.zeros((len(xyz_positions[:, 0]), 3, n_channels))
    # fill in the antenna id's
    gain_table[:, 0, :] = numpy.array([xyz_positions[:, 0], ] * n_channels).transpose()
    # set the antenna amplitude gains
    gain_table[:, 1, :] = 1.
    # set the antenna phases
    gain_table[:, 2, :] = 0.

    return gain_table


def antenna_table_loader(path):
    print path
    antenna_data = numpy.loadtxt(path)

    # sort the antenna based on x- and y-coordinates.
    antenna_data = antenna_data[numpy.argsort(antenna_data[:, 1])]
    antenna_data = antenna_data[numpy.argsort(antenna_data[:, 2])]
    return antenna_data
