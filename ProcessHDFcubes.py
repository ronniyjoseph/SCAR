from CubeProcessing import data_processor


def main():
    output_path ="/home/rjoseph/Bulk/Redundant_Calibration/Simulation_Output/TEST0_SFPO_Hex_P_BG_full_NEW/"
    output_path = "/Users/ronniyjoseph/Sync/PhD/Projects/redundant_calibration/simulation_output/" \
              "TEST0_SFPO_Hex_P_BG_full_FIX/"
    simulation_type = "CRAMPS"
    simulation_type ="SFPO"


    stacking_mode = [True]
    histogram_plotting = [True, 'noisy']# "both"
    averaging_type = [False,"median",'std'] #"median" or "mean"

    data_processor(output_path, simulation_type, stacking_mode, histogram_plotting, averaging_type)




if __name__ == "__main__":
    main()
