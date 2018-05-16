from CubeProcessing import data_processor


def main():
    output_path ="/home/rjoseph/Bulk/Redundant_Calibration/Simulation_Output/CRAMPS_Linear_ideal_P1000_BG_fullcal/"
    simulation_type = "CRAMPS"

    #output_path = "/home/rjoseph/Bulk/Redundant_Calibration/Simulation_Output/TEST2_SLPO_HEX_P_BG_Logcal_NEW/"
    #simulation_type ="SLPO"

    stacking_mode = [True]
    histogram_plotting = [False,'both']# "both"
    averaging_type = [True,"median",'std'] #"median" or "mean"

    data_processor(output_path, simulation_type, stacking_mode, histogram_plotting, averaging_type)




if __name__ == "__main__":
    main()
