from CubeProcessing import cube_processor


def main():
    output_path = "../../simulation_output/"
    simulation_run = "CRAMPS_Linear_P_BG_Offset_logcal"
    simulation_type = "CRAMPS" \
                      ""
    histogram_plotting =  [False,"both"]
    histogram_plotset = "both" #ideal noisy or both
    solution_averaging =  False
    averaging_type = [True,"median",'std'] #"median" or "mean"
    variance_type = "std" #variance type "std" or "iqr"

    cube_processor(output_path,simulation_run,simulation_type,histogram_plotting, averaging_type)




if __name__ == "__main__":
    main()