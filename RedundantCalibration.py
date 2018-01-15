from GeneralTools import unique_value_finder
from GeneralTools import position_finder

import numpy


def LogcalMatrixPopulator(uv_positions, xyz_positions):
    # so first we sort out the unique antennas
    # and the unique redudant groups, this will allows us to populate the matrix adequately
    red_tiles = unique_value_finder(uv_positions[:, 0:2], 'values')
    # it's not really finding unique antennas, it just finds unique values
    red_groups = unique_value_finder(uv_positions[:, 7], 'values')
   # print "There are", len(red_tiles), "redundant tiles"
   # print ""
   # print "Creating the equation matrix"
    # create am empty matrix (#measurements)x(#tiles + #redundant groups)
    amp_matrix = numpy.zeros((len(uv_positions), len(red_tiles) + len(red_groups)))
    phase_matrix = numpy.zeros((len(uv_positions), len(red_tiles) + len(red_groups)))
    for i in range(len(uv_positions)):
        index1 = numpy.where(red_tiles == uv_positions[i, 0])
        index2 = numpy.where(red_tiles == uv_positions[i, 1])
        index_group = numpy.where(red_groups == uv_positions[i, 7])

        amp_matrix[i, index1[0]] = 1
        amp_matrix[i, index2[0]] = 1
        amp_matrix[i, len(red_tiles) + index_group[0]] = 1

        phase_matrix[i, index1[0]] = 1
        phase_matrix[i, index2[0]] = -1
        phase_matrix[i, len(red_tiles) + index_group[0]] = 1

    # select the xy-positions for the red_tiles
    red_x_positions, red_y_positions = position_finder(red_tiles, \
                                                       xyz_positions)

    # add this to the amplitude matrix
    amp_constraints = numpy.zeros((len(red_tiles) + len(red_groups)))
    amp_constraints[0] = 1.
    amp_matrix = numpy.vstack((amp_matrix, amp_constraints))

    # add these constraints to the phase matrix
    phase_constraints = numpy.zeros((3, len(red_tiles) + len(red_groups)))
    phase_constraints[0, 0] = 1
    phase_constraints[1, 0:len(red_tiles)] = red_x_positions
    phase_constraints[2, 0:len(red_tiles)] = red_y_positions
    phase_matrix = numpy.vstack((phase_matrix, phase_constraints))

    # check whether the matrix is ill conditioned
    phase_dagger = numpy.dot(phase_matrix.transpose(), phase_matrix)
    amp_dagger = numpy.dot(amp_matrix.transpose(), amp_matrix)
    if numpy.linalg.det(numpy.dot(numpy.linalg.pinv(amp_dagger), amp_dagger)) == 0:
        print "WARNING: the amplitude solver matrix is singular"

    if numpy.linalg.det(numpy.dot(numpy.linalg.pinv(phase_dagger), phase_dagger)) == 0:
        print "WARNING: the phase solver matrix is singular"

    phase_pinv = numpy.dot(numpy.linalg.pinv(phase_dagger), phase_matrix.transpose())
    amp_pinv = numpy.dot(numpy.linalg.pinv(amp_dagger), amp_matrix.transpose())

    return amp_pinv, phase_pinv, red_tiles, red_groups


def LogcalSolver(amp_pinv, phase_pinv, obs_visibilities):
    # print "Preparing to solve for",len(amp_matrix[0,:]),"antennas and", \
    # len(red_groups),"unique visibilities."
    # now we can set up the logcal solver algorithm
    ########## Amplitudes ####################

    #########################################
    # take the log of the measured visibilities
    amp_log = numpy.zeros((len(obs_visibilities) + 1))
    amp_log[0:len(obs_visibilities)] = numpy.log( \
        numpy.absolute(obs_visibilities))
    # Set the reference antenna amplitude to 1
    amp_log[len(amp_log) - 1] = numpy.log(1)

    ################# Phase ########################
    phase_log = numpy.zeros(len(obs_visibilities) + 3)
    phase_log[0:len(obs_visibilities)] = numpy.angle(obs_visibilities)

    # add the reference antenna, constraint,
    # and the tilt constraints
    phase_log[len(phase_log) - 3:len(phase_log) - 1] = 0.

    # print ""
    # print ""
    # ~ print ""
    # ~ print "Solving for the amplitude and phase gains"
    amp_solutions = numpy.dot(amp_pinv, amp_log)
    phase_solutions = numpy.dot(phase_pinv, phase_log)

    return numpy.exp(amp_solutions), phase_solutions


def LincalMatrixPopulator(uv_positions, correlation_obs, gain_0, visibility_0, red_tiles, red_groups):
    A_matrix = numpy.zeros((2 * len(uv_positions), 2 * len(red_tiles) + 2 * len(red_groups)))
    d_correlation = numpy.zeros(2 * len(uv_positions))

    for i in range(len(uv_positions)):
        # index of the antennas and find their gains
        index1 = numpy.where(red_tiles == uv_positions[i, 0])[0]
        index2 = numpy.where(red_tiles == uv_positions[i, 1])[0]

        # index the redundant group and find its value
        index_group = numpy.where(red_groups == uv_positions[i, 7])[0]

        correlation_0 = gain_0[index1] * numpy.conj(gain_0[index2]) * visibility_0[index_group]
        d_correlation[2 * i] = numpy.real(correlation_obs[i] - correlation_0)
        d_correlation[2 * i + 1] = numpy.imag(correlation_obs[i] - correlation_0)

        # Fill in the real row
        A_matrix[2 * i, 2 * index1] = numpy.real(numpy.conj(gain_0[index2]) * visibility_0[index_group])
        A_matrix[2 * i, 2 * index1 + 1] = -numpy.imag(numpy.conj(gain_0[index2]) * visibility_0[index_group])
        A_matrix[2 * i, 2 * index2] = numpy.real(gain_0[index1] * visibility_0[index_group])
        A_matrix[2 * i, 2 * index2 + 1] = numpy.imag(gain_0[index1] * visibility_0[index_group])

        A_matrix[2 * i, 2 * (len(red_tiles) + index_group)] = numpy.real(gain_0[index1] * numpy.conj(gain_0[index2]))
        A_matrix[2 * i, 2 * (len(red_tiles) + index_group) + 1] = -numpy.imag(
            gain_0[index1] * numpy.conj(gain_0[index2]))
        # Fill in the imaginary row
        A_matrix[2 * i + 1, 2 * index1] = numpy.imag(numpy.conj(gain_0[index2]) * visibility_0[index_group])
        A_matrix[2 * i + 1, 2 * index1 + 1] = numpy.real(numpy.conj(gain_0[index2]) * visibility_0[index_group])
        A_matrix[2 * i + 1, 2 * index2] = numpy.imag(gain_0[index1] * visibility_0[index_group])
        A_matrix[2 * i + 1, 2 * index2 + 1] = -numpy.real(gain_0[index1] * visibility_0[index_group])

        A_matrix[2 * i + 1, 2 * (len(red_tiles) + index_group)] = numpy.imag(
            gain_0[index1] * numpy.conj(gain_0[index2]))
        A_matrix[2 * i + 1, 2 * (len(red_tiles) + index_group) + 1] = numpy.real(
            gain_0[index1] * numpy.conj(gain_0[index2]))

    # Calculate the inverse matrix
    # check whether the matrix is ill conditioned
    A_dagger = numpy.dot(A_matrix.transpose(), A_matrix)
    if numpy.linalg.det(numpy.dot(numpy.linalg.pinv(A_dagger), A_dagger)) == 0:
        print "WARNING: the Lincal solver matrix is singular"
    A_pinv = numpy.dot(numpy.linalg.pinv(A_dagger), A_matrix.transpose())

    return A_pinv, d_correlation


def LincalSolver(uv_positions, correlation_obs, amp_solutions, phase_solutions, red_tiles, red_groups):
    gain_0 = amp_solutions[:len(red_tiles)] * numpy.exp(1j * phase_solutions[:len(red_tiles)])
    visibility_0 = amp_solutions[len(red_tiles):] * numpy.exp(1j * phase_solutions[len(red_tiles):])

    # ~ fig = pyplot.figure()#figsize=(5,5))
    # ~ errorsub = fig.add_subplot(1,2,1)
    # ~ diffsub = fig.add_subplot(1,2,2)

    # ~ errorsub.set_title(r'$g - g_{true}$')
    # ~ diffsub.set_title(r'$g_{i} - g_{i - 1}$')

    convergence = False

    counter = 0
    d_corrections = 1
    while d_corrections > 1e-9 and counter < 1000:

        A_pinv, d_correlation = LincalMatrixPopulator(uv_positions, correlation_obs, gain_0, visibility_0, red_tiles,
                                                      red_groups)
        d_solutions_1 = numpy.dot(A_pinv, d_correlation)

        error = numpy.sum(numpy.abs(d_correlation)) / len(d_correlation)
        # ~ errorsub.plot(counter,error,"r+")

        gain_1 = gain_0 + (d_solutions_1[0:2 * len(red_tiles):2] + 1j * d_solutions_1[1:2 * len(red_tiles) + 1:2])
        visibility_1 = visibility_0 + (
                d_solutions_1[2 * len(red_tiles)::2] + 1j * d_solutions_1[2 * len(red_tiles) + 1::2])

        # calculate difference
        if counter > 0:
            d_corrections = numpy.sum(numpy.abs((d_solutions_1 - d_solutions_0) ** 2))  # /numpy.sum(abs(d_solutions_1))
        # ~ diffsub.plot(counter,d_corrections,"bx")

        d_averaged_1 = numpy.sum(d_solutions_1 * numpy.conjugate(d_solutions_1))

        counter += 1

        gain_0 = gain_1.copy()
        visibility_0 = visibility_1.copy()
        d_solutions_0 = d_solutions_1.copy()
        # print "Lincal required %d iterations to converge" %counter
    return numpy.hstack((gain_1, visibility_1))


def Redundant_Calibrator(amp_matrix, phase_matrix, obs_visibilities,
                         red_baseline_table, red_tiles, red_groups, calibration_scheme):
    ####################Redundant Calibration###########################
    if calibration_scheme[0] == 'logcal':
        # feed observations into a gain solver function
        amp_solutions, phase_solutions = \
            LogcalSolver(amp_matrix, phase_matrix, obs_visibilities[:, 0])
    elif calibration_scheme[0] == 'lincal':
        true_solutions = calibration_scheme[1]

        amp_guess = numpy.abs(true_solutions)
        phase_guess = numpy.abs(true_solutions)

        lincal_solutions = LincalSolver(red_baseline_table, obs_visibilities,
                                        amp_guess, phase_guess, red_tiles,
                                        red_groups)
        amp_solutions = numpy.abs(lincal_solutions)
        phase_solutions = numpy.angle(lincal_solutions)
    elif calibration_scheme[0] == 'full':
        amp_guess, phase_guess = \
            LogcalSolver(amp_matrix, phase_matrix, obs_visibilities[:, 0])
        lincal_solutions = LincalSolver(red_baseline_table, obs_visibilities,
                                        amp_guess, phase_guess, red_tiles, red_groups)

        amp_solutions = numpy.abs(lincal_solutions)
        phase_solutions = numpy.angle(lincal_solutions)
    ####################Absolute Gain Scaler#######################

    # ~ #solve for the overall gain and rescale
    # ~ amp_visibilities,amp_gains = absolute_amplitude_rescaler(\
    # ~ obs_visibilities,source_visibilities, amp_visibilities,\
    # ~ amp_gains,abs_amp_matrix,redundant_positions)

    return amp_solutions, phase_solutions
