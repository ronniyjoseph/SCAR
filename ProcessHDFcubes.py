from CubeProcessing import cube_processor


def main():
    output_path = "/home/rjoseph/Bulk/Redundant_Calibration/Simulation_Output/"

    simulation_run = "CRAMPS_G_Linear_P200_Offset2_0.15_logcal"
    simulation_type = "CRAMPS"

    #simulation_run = "SiSpS_HDF5_P_Linear_Iter1e3"
    #simulation_type = ["SiSpS", "SFPO"]


    histogram_plotting = [False,'both']# "both"
    averaging_type = [True,"median",'std'] #"median" or "mean"
    variance_type = "std" #variance type "std" or "iqr"

    cube_processor(output_path,simulation_run,simulation_type, histogram_plotting, averaging_type)




if __name__ == "__main__":
    main()
