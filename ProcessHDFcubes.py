from CubeProcessing import cube_processor


def main():
    output_path = "../../simulation_output/"
    simulation_run = "SiSpS_HDF5_Test"
    simulation_type = "SiSpS" \
                      ""
    histogram_plotting =  True
    histogram_plotset = "both" #ideal noisy or both
    solution_averaging =  False
    averaging_type = "median" #"median" or "mean"
    variance_type = "std" #variance type "std" or "iqr"

    cube_processor(output_path,simulation_run,simulation_type,histogram_plotset,solution_averaging)




if __name__ == "__main__":
    main()