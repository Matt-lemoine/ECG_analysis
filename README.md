This is the README.md for the EKG_analysis pipeline. I have outlined below all the different files in this repo, the functions in these files, what they do, and how to use them. There is also more documentation in each of the files outlining how the functions call and work.

In this repo we will have a pipeline that takes in EKG readings and outputs
    Persistent homology of the SWE of the EKG.

Visualize_EKG.py info:
    1. get_EKG_leads(ptf, lead_s)
        Input:
            ptf = Path to file, this is the path from the file you are currently in to the desired EKG, you'd like to analyze.
            lead_s = these are the leads you are interested in. This entry should be an array of strings looking like the following.
                All possible lead_s = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
                From this big list, you can specify which leads you are analyzing.
                Example lead_s = ['I', 'III', 'V3']
        Output:
            p_signal = This is a numpy array with the leads you are interested in as a matrix. The matrix will be (t*100 x len(lead_s))
                where t is time. So if your EKG is recorded for 12 seconds and you are interested in only 4 of the leads, your output will
                be an np.array of dimension (1200 x 4).
        This function takes in the path to the file for the EKG with the leads that you want to focus on, and returns the leads you are
            interested in as an numpy array. These leads can then be passed to Wavelet Approximation, to approximate the wavelet.
    2. plot_ekg(ptf, lead_s, naming_things) 
        Input: 
            ptf = Path to file, this is the path from the working folder to the folder with the .dat and .hea files.
            lead_s = These are the leads you are interested in plotting.
            naming_things = This is what you want to name the plot.
        Output:
            The output is a matplotlib figure with the plots you are interested in.
    3. get_all_info(ptf)
        Input:
            ptf = Path to file, this is the path from the working folder to the folder with the .dat and .hea files.
        Output:
            The output of this function is the same as the output of the '__dict__' function. It tells you everything about your data
            from the .dat and .hea files.
Wavelet_approx.py info:
    1. plot_wavelet(ptf, lead_s, wavelet, level_decomp, naming_things, sample_num = None)
    2. wavelet_coeffs(ptf, lead_s, wavelet, level_decomp)
    3. plot_wavelet_w_coeffs(ptf, lead_s, wavelet, level_decomp)
    4. wavelet(ptf, lead_s, wavelet, level_decomp, t)
        This is the big one. It plots an approximate value at time t for your wavelet approximation of your EKG.

SWE.py info:

main_EKG.py info: