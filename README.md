In this repo we will have a pipeline that takes in EKG readings and outputs
    Persistent homology of the SWE of the EKG.

Visualize_EKG info:
    1. get_EKG_leads(ptf, lead_s)

        Input:
            ptf = Path to file, this is the path from the file you are currently in to the desired EKG, you'd like to analyze.
            lead_s = these are the leads you are interested in. This entry should be an array of strings looking like the following.
                Example lead_s = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
                From this big list, you can specify which leads you are analyzing.
                Example lead_s = ['I', 'III', 'V3']
        Output:
            p_signal = This is a numpy array with the leads you are interested in as a matrix. The matrix will be (t*100 x len(lead_s))
                where t is time. So if your EKG is recorded for 12 seconds and you are interested in only 4 of the leads, your output will
                be an np.array of dimension (1200 x 4).