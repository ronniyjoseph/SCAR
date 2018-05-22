from CubeProcessing import data_processor


def main():
    output_path ="/home/rjoseph/Bulk/Redundant_Calibration/Simulation_Output/TEST2_SFPO_HEX_P_BG_Logcal_FIX/"
    simulation_type = "SFPO"
    #simulation_type ="SLPO"


    stacking_mode = [False]
    histogram_plotting = [True,'both']# "both"
    averaging_type = [False,"median",'std'] #"median" or "mean"

    data_processor(output_path, simulation_type, stacking_mode, histogram_plotting, averaging_type)




if __name__ == "__main__":
    main()
