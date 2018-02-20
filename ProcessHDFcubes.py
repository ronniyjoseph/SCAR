from CubeProcessing import cube_processor


def main():
    output_path = "../../simulation_output/"

    simulation_run = "CRAMPS_P10_Linear4"
    simulation_type = "CRAMPS"

    #simulation_run = "SiSpS_HDF5_large"
    #simulation_type = ["SiSpS", ""]


    histogram_plotting = [False,'both']# "both"
    averaging_type = [True,"median",'std'] #"median" or "mean"
    variance_type = "std" #variance type "std" or "iqr"

    cube_processor(output_path,simulation_run,simulation_type, histogram_plotting, averaging_type)




if __name__ == "__main__":
    main()
