import argparse
from Multi_SimRunner import source_location_changer_MP
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


def main(n_channels, output_folder, source_strength, calibration_scheme, multi_processing, off_bool):
    calibration_channel = [150e6]
    channel_size = 40e3
    sky_steps = 666
    sky_param = ['point_and_background', source_strength]     #Note with single point source noise will be point 10% point source dominated
    noise_param = ['SEFD', 20e3, 40e3, 120]
    beam_param = ['gaussian', 0.25, 0.25]
    source_position_range = [-1, 1, 777]
    direction = 'l'
    make_histogram_movie = False
    iterations = 999
    save_to_disk = [True, output_folder]
    telescope_param = ["linear", 10, 5]
    #telescope_param = ["hex", 14., 0, 0]
    hist_movie = [make_histogram_movie, output_folder]
    offset_param = [off_bool, 2,'x', 0.15]

    if n_channels == 0:
        if multi_processing[0]:
            source_location_changer_MP(telescope_param, offset_param, calibration_channel, noise_param, direction,
                  sky_steps, source_position_range, iterations, sky_param, beam_param, calibration_scheme, save_to_disk,
                                       multi_processing[1])
        else:
            Moving_Source(telescope_param, offset_param, calibration_channel, noise_param, 'l',
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
    parser.add_argument('-path', action='store',
                        default="/home/rjoseph/Bulk/Redundant_Calibration/Simulation_Output/"
                                                         "TEST_CRAMPS_PBG/",
                        type=str)
    parser.add_argument('-cal_mode', action='store', default='logcal', type=str)
    parser.add_argument('-MP', action='store_true', default=False,)
    parser.add_argument('-MP_Processes',  action='store', default=8,
                        type=int)
    parser.add_argument('-offset', action='store_true', default=False,)
    args = parser.parse_args()

    main(args.Nchan, args.path, args.pointJy, args.cal_mode,[args.MP, args.MP_Processes], args.offset)

