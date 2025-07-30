
Mental-pong data from monkey Dorso-Medial Frontal Cortex (DMFC)
=============
    Dynamic tracking of objects in the macaque dorsomedial frontal cortex

    Rishi Rajalingham*, Hansem Sohn*, Mehrdad Jazayeri (2024)

##### Folder structure:

    |-- MentalPong
    |   |-- code
    |       |-- utils
    |   |-- data: in 10.5281/zenodo.13952210
    |   |-- analyses
    |       |-- behavior
    |       |-- single_neuron_encode
    |       |-- population_decode
    |           |-- online
    |           |-- offline
    |       |-- rnn
    |       |-- shell_scripts

##### Details:

    |-- MentalPong/code
      
Contains codes that generate figures in the paper. File names indicate corresponding figures. Codes are either jupyter notebook or matlab live scripts. Auxiliary codes that support the figure generation is in 'utils' folder.
    
    |-- MentalPong/data
    
Contains source data for all figures ('Source_Data.xlsx'). All other raw or processed data for figure generations are saved either in python pickle or matlab data format. Due to storage limit, it is in 10.5281/zenodo.13952210
    
   - binned neural data: bin size = 50ms, 79 task conditions
     - python pickle: 
          - 'all_hand_dmfc_dataset_50ms.pkl' : pooled across two animals
          - 'perle_hand_dmfc_dataset_50ms.pkl': 1st monkey
          - 'mahler_hand_dmfc_dataset_50ms.pkl': 2nd monkey
          - 'random_dataset_50ms.pkl': shuffled control dataset
            > e.g., initial ball position and velocity: ['behavioral_responses']['occ']['ball_pos_x']. The second key is for the occluded epoch and can be ['vis'] for the visible epoch. For velocity, the last key should be ['ball_pos_dx']. There it's 79 x 100 matrix, where 79 corresponds to number of conditions and 100 are for time points (assuming 50ms bin size). Length of each epoch differs across conditions, which can be found in ['t_from_start'] and ['t_from_occ'].
     - matlab for single neuron analysis: 'perle_hand_dmfc_dataset_50ms.mat', 'mahler_hand_dmfc_dataset_50ms.mat', 'all_hand_dmfc_dataset_50ms.mat' (pooled across two animals)
   - single neuron encoding analysis by generalized linear modeling & variance partitioning analysis: 'run_glm.mat', 'run_glm_egocentric_ball.mat', 'PVAF_allocentric.mat' (the last two are for comparing egocentric vs. allocentric models)
   - population-level decoding analysis: 
     - 'decode_all_hand_dmfc_occ_start_end_pad0_50ms_0.50_neural_responses_reliable_FactorAnalysis_50.pkl': pooled across two animals
     - 'decode_perle_hand_dmfc_occ_start_end_pad0_50ms_0.50_neural_responses_reliable_FactorAnalysis_50.pkl': 1st monkey
     - 'decode_mahler_hand_dmfc_occ_start_end_pad0_50ms_0.50_neural_responses_reliable_FactorAnalysis_50.pkl': 2nd monkey
     - 'decode_all_hand_dmfc_ego_occ_start_end_pad0_50ms_0.50_neural_responses_reliable_FactorAnalysis_50.pkl': decoding for egocentric model
   - comparison between RNN and DMFC using representational similarity analysis: 'rnn_compare_all_hand_dmfc_occ_50ms_neural_responses_reliable_FactorAnalysis_50.pkl'
   - offline
        - 'offline_all_hand_dmfc_neural_responses_reliable_50.pkl': pooled across two animals
        - 'offline_perle_hand_dmfc_neural_responses_reliable_50.pkl': 1st monkey
        - 'offline_mahler_hand_dmfc_neural_responses_reliable_50.pkl': 2nd monkey
        - 'offline_all_hand_dmfc_neural_responses_reliable_50_occ.pkl': decoding from time of occlusion start
        - 'offline_rnn_neural_responses_reliable_50.pkl': RNN
        
            > In 'offline_rnn_neural_responses_reliable_50.pkl', there are three variables:    
            > 1. 'all_metrics': dictionary that contains 192 RNN models' endpoint prediction
            >   - 'yp': predicted ball y position, ndarray (100,79,90) (# train/test splits by GroupShuffleSplit, # conditions, # timepoints)
            >   - 'yt': true ball y position, ndarray (100,79,90) (# train/test splits by GroupShuffleSplit, # conditions, # timepoints)
            >   - other variables (e.g., quantifying decoding performance using correlation, r or mae)                
            > 2. 'state': dictionary that contains 192 RNN models' hidden states
            >   - 'data['data_neur_nxcxt']': hidden states, ndarray (10 or 20 or 40,79,90) (# units, # conditions, # timepoints)
            >   - 'data['data_beh_bxcxt']': ball_y_final for decoding target, ndarray (1,79,90) (1, # conditions, # timepoints)   
            >   - other variables               
            > 3. 'df': DataFrame that contains all detailed information about 192 RNN models [192 rows x 695 columns]
        
   - 'valid_meta_sample_full.pkl': contains task condition information

    |-- MentalPong/analyses
    
All analysis codes are put here afer being sorted into different categories: 

  - behavior
  - single neuron encoding analysis
  - population decoding analysis
  - recurrent neural network (rnn)
  - miscellaneous shell scripts. 
  
Note that these analysis codes are not edited from original versions so may not run on its own (e.g., you may need to change directory address to point relevant data). Contact authors if you need help.

Modifications and Contributions by Jamyaha Cleckley

As part of my research internship, I expanded this project to include:

    Baseline Prediction Analysis (see newfigure3ef(2).ipynb):
    Implemented a simple baseline model that predicts average ball position across trials, and calculated RMSE over time to benchmark decoding performance.

    Decoder Performance Analysis (see simplepredfig.ipynb):
    Added condition-wise comparisons (bounce vs non-bounce trials), separated RMSE evaluations, and visualized how neural decoders perform relative to baselines.

    Utilities:

        Created or modified utility functions in decoding_summary.py and dataset_augment_utils.py for plotting, masking, and RMSE calculation.

These updates help quantify how well neural activity in DMFC predicts object motion during occlusion, supporting the role of mental simulation in decision making and motor planning.
