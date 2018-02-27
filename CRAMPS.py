import argparse
from SimRunner import MuChSource_Mover
from SimRunner import Moving_Source
import numpy

########################################################################
#   Calibrate Redundant Arrays on Moving Point Sources                 3
########################################################################
# This code calculates the average estimated gains (phase/amplitude) for
# an array given its antenna positions. Assuming ideal amplitude gain =1
# and phase gain = 0.
# Loads antenna positions, calculates corresponding uv positions and then
# selects the redundant baselines. For these baselines the measured
# fourier correlations of a single source sky are calculated based on the
# input gains. Which can be set to ideal or non ideal. Gaussian Noise is
# added to these correlations before they are processed through the
# logcal machinery of redundant calibration.
# 20-01-2017
# Added random background sky and one strong source


def main(n_channels, output_folder, source_strength):
    calibration_channel = 150e6
    channel_size = 40e3
    sky_steps = 999
    sky_param = ['point_and_background', source_strength]     #Note with single point source noise will be point 10% point source dominated
    noise_param = ['SEFD', 20e3, 40e3, 120]
    beam_param = ['gaussian', 0.25, 0.25]
    make_histogram_movie = False
    iterations = 1001
    save_to_disk = [True, output_folder]
    telescope_param = ["linear", 10, 5]
    hist_movie = [make_histogram_movie, output_folder]
    calibration_scheme = 'logcal'  # 'logcal','lincal','full'
    offset = [True, 2,'x', 0.15]

    tile_numbers = numpy.arange(0,5,1)
    shifts = numpy.random.normal(0,0.18,5)
    offset = [True, tile_numbers,'x', shifts]

    if n_channels == 0:
        Moving_Source(telescope_param, offset, calibration_channel, noise_param, 'l',
                      sky_steps, iterations, sky_param, beam_param, calibration_scheme, save_to_disk, hist_movie)
    elif n_channels > 0:
        MuChSource_Mover(n_channels, telescope_param, calibration_channel, noise_param, 'l',
                         sky_steps, sky_param, beam_param, save_to_disk)

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Redundant \
	 Calibration Simulation set up')
    parser.add_argument('-Nchan', action='store', default=0,
                        type=int)
    parser.add_argument('-pointJy', action='store', default=200,
                        type=float)
    parser.add_argument('-path', action='store', default="../../simulation_output/CRAMPS_P200_Offset_All_Random/",
                        type=str)

    args = parser.parse_args()

    main(args.Nchan, args.path, args.pointJy)
