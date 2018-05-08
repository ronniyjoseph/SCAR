import argparse
from Multi_Fixed_SimRunner import source_location_and_position_offset_changer_FixedMP
from Multi_Fixed_SimRunner import source_flux_and_position_offset_changer_FixedMP
from Multi_SimRunner import source_flux_and_position_offset_changer_parallel
from Multi_SimRunner import source_location_and_position_offset_changer_parallel
from SimRunner import source_flux_and_position_offset_changer as source_flux_and_position_offset_changer_serial
from SimRunner import moving_source_and_position_offset_changer as moving_source_and_position_offset_changer_serial

########################################################################
# Calls the simulation which varies position offsets and peak fluxes   #
########################################################################

def main(output_folder,sim_type, fixed_array, multi_processing):
    #multi_processing = [True, 8]
    calibration_channel = [150e6]
    channel_size = 40e3
    sky_param = ['point_and_background', 200, 0.05, 0.]
    #sim_type = "changing_flux"
    noise_param = ['SEFD', 20e3, 40e3, 120]
    beam_param = ['gaussian', 0.25, 0.25]
    #fixed_array = True
    iterations = 999
    peakflux_range = [1, 2e2, 49]    #Specify in Jy
    offset_range = [1e-4, 0.5, 51]  #Specify in m
    source_position_range = [-1, 1, 555]
    save_to_disk = [True, output_folder]
    #telescope_param = ["hex", 14., 0, 0]
    telescope_param = ["linear", 10, 5, 0]

    calibration_scheme = 'logcal'

    if sim_type == "changing_flux":
        if multi_processing[0]:
            if fixed_array:
                source_flux_and_position_offset_changer_FixedMP(telescope_param, calibration_channel, noise_param,
                                                                sky_param, beam_param,
                                                                calibration_scheme, peakflux_range, offset_range,
                                                                iterations,
                                                                save_to_disk,
                                                                multi_processing[1])
            else:
                source_flux_and_position_offset_changer_parallel(telescope_param, calibration_channel, noise_param, sky_param,
                                                    beam_param, calibration_scheme, peakflux_range, offset_range,
                                                    iterations,
                                                    save_to_disk, multi_processing[1])

        else:
            source_flux_and_position_offset_changer_serial(telescope_param, calibration_channel, noise_param, sky_param,
                                                    beam_param, calibration_scheme, peakflux_range, offset_range,
                                                    iterations,
                                                    save_to_disk)
    elif sim_type == "moving_source":
        if multi_processing[0]:
            if fixed_array:
                source_location_and_position_offset_changer_FixedMP(telescope_param, calibration_channel, noise_param,
                                                                sky_param, beam_param,
                                                                calibration_scheme, peakflux_range, offset_range,
                                                                iterations,
                                                                save_to_disk,
                                                                multi_processing[1])
            else:
                source_location_and_position_offset_changer_parallel(telescope_param, calibration_channel, noise_param, sky_param,
                                                     beam_param, calibration_scheme, source_position_range, offset_range,
                                                     iterations, save_to_disk, multi_processing[1])
        else:
            moving_source_and_position_offset_changer_serial(telescope_param, calibration_channel, noise_param, sky_param,
                                                  beam_param, calibration_scheme, source_position_range, offset_range,
                                                  iterations,save_to_disk)
    else:
        sys.exit("Wrong simulation choice....")
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Redundant \
     Calibration Simulation set up')
    parser.add_argument('-path', action='store',
                        default="/home/rjoseph/Bulk/Redundant_Calibration/Simulation_Output/"
                                                         "TEST2_SFPO_Linear_P_BG_Logcal_NEW/",
                        type=str)
    parser.add_argument('-sim_type',  action='store', default="moving_source",
                        type=str)
    parser.add_argument('-array_fix', action='store_true', default=False)
    parser.add_argument('-MP', action='store_true', default=False,)
    parser.add_argument('-MP_Processes',  action='store', default=8,
                        type=int)


    args = parser.parse_args()
    main(args.path, args.sim_type, args.array_fix, [args.MP, args.MP_Processes])
