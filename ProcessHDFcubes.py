from CubeProcessing import data_processor


def main():
    output_path = "/home/rjoseph/Bulk/Redundant_Calibration/Simulation_Output/"

    simulation_run = "CRAMPS_G_Linear_P200BG_Ideal_OffsetAll_Random_logcal"
    simulation_type = "CRAMPS"

    output_path = "/home/rjoseph/Bulk/Redundant_Calibration/Simulation_Output/SiSpS_Linear_P_BG/"
    simulation_type ="SFPO"

    stacking_mode = [False]
    histogram_plotting = [True,'both']# "both"
    averaging_type = [False,"median",'std'] #"median" or "mean"

    data_processor(output_path, simulation_type, stacking_mode, histogram_plotting, averaging_type)




if __name__ == "__main__":
    main()
