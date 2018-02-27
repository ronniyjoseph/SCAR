import argparse
from SimRunner import source_flux_and_position_offset_changer
from SimRunner import moving_source_and_position_offset_changer

########################################################################
# Calls the simulation which varies position offsets and peak fluxes   #
########################################################################

def main(n_channels, output_folder, source_strength):
    calibration_channel = [150e6]
    channel_size = 40e3
    sky_param = ['point', 100., 0.1, 0.]
    sim_type = "changing_flux"
    noise_param = ['SEFD', 20e3, 40e3, 120]
    beam_param = ['gaussian', 0.25, 0.25]
    iterations = 100000
    peakflux_range = [1, 2e2,  49]    #Specify in Jy
    offset_range = [1e-4, 0.15, 51]  #Specify in m
    save_to_disk = [True, output_folder]
    #telescope_param = ["hex", 14., 0, 0]
    telescope_param = ["linear", 10, 5, 0]

    calibration_scheme = 'logcal'

    if sim_type == "changing_flux":
        source_flux_and_position_offset_changer(telescope_param, calibration_channel, noise_param,sky_param,
                                           beam_param, calibration_scheme, peakflux_range, offset_range, iterations,
                                           save_to_disk)
    elif sim_type == "moving_source":
        moving_source_and_position_offset_changer(telescope_param, calibration_channel, noise_param, sky_param,
                                                  beam_param, calibration_scheme,offset_range,
                                                  iterations,save_to_disk)
    else:
        sys.exit("Wrong simulation choice....")
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Redundant \
     Calibration Simulation set up')
    parser.add_argument('-Nchan', action='store', type=int,
                        default=0)
    parser.add_argument('-pointJy', action='store', default=200.,
                        type=float)
    parser.add_argument('-path', action='store', default="../../simulation_output/SiSpS_HDF5_P_Linear_Iter1e5/",
                        type=str)

    args = parser.parse_args()

    main(args.Nchan, args.path, args.pointJy)
