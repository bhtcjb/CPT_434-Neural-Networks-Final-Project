import pandas as pd
import glob
import os
import mne # pip install mne-lsl
import mne_nirs # pip install mne-nirs
import numpy as np
import matplotlib.pyplot as plt


def convert_fNIRS_to_raw(fNIRS_df):
    
    # get pure data
    fNIRS_data = fNIRS_df.drop(columns=['ts']).values.T

    # the naming system for fNIRS uses S#_D# ### where S is source and D is detector
    fNIRS_ch_names = [
        'S1_D1 730', 'S2_D3 730',  # left outer 730, right outer 730
        'S1_D1 850', 'S2_D3 850',  # left outer 850, right outer 850
        'S1_D2 730', 'S2_D4 730',  # left inner 730, right inner 730
        'S1_D2 850', 'S2_D4 850',  # left inner 850, right inner 850
    ]  
    
    # optic freq is 64 Hz according to documentation
    # optic is in µA which are arbitrary units based on luminance flux
    # type set to 'fnirs_cw_amplitude' continuous wave fNIRS for light intensity
    fNIRS_info = mne.create_info(ch_names=fNIRS_ch_names, sfreq=64, ch_types='fnirs_cw_amplitude')
    
    ### manually setting metadata for the info object because i couldn't figure out how to set up the montage for correct 3d topo mappings

    # x coordinate, represents left right - units in meters, 9.5 cm is default radius of topographic head model
    x_mappings = [-0.03,  0.03, -0.03,  0.03, -0.01,  0.01, -0.01,  0.01]
    
    # y coordinate, represents front back - set 1.5 cm inside head
    y_mappings = [ 0.08,  0.08,  0.08,  0.08,  0.08,  0.08,  0.08,  0.08]
    
    # z coordinate, represents up down
    z_mappings = [ 0.02,  0.02,  0.02,  0.02,  0.02,  0.02,  0.02,  0.02]
    
    # Note that for the above coordinate mappings matter for visualization only, and so only need to correspond to the channel
    # lengths so far in as you want the visualization to look realistic. 
    # However, he channel lengths do matter for data calculations and should match the device specifications for accurate results
    
    # channel lengths are 3 cm for long and 1 cm for short
    distances = [0.03, 0.03, 0.03, 0.03, 0.01, 0.01, 0.01, 0.01]
    
    # label wavelengths - units in nanometers
    wavelengths = [730, 730, 850, 850, 730, 730, 850, 850]

    # apply all metadata to the info object
    for i, chs in enumerate(fNIRS_info['chs']):
        
        # some operations don't work if unused values are not initialized
        loc = np.zeros(12)
        
        loc[0] = x_mappings[i]
        loc[1] = y_mappings[i]     
        loc[2] = z_mappings[i]     
        loc[3] = distances[i]
        loc[9] = wavelengths[i]
        chs['loc'] = loc
       
        
    # convert to MNE raw type to be processed
    fNIRS_raw = mne.io.RawArray(data=fNIRS_data, info=fNIRS_info)
    
    return fNIRS_raw

def convert_EEG_to_raw(EEG_df):
    
    # get pure data
    EEG_data = EEG_df.drop(columns=['ts']).values.T * 1e-6 # convert to V
    
    # left temporal, left frontal, right frontal, right temporal
    EEG_ch_names = [ 'TP9', 'Fp1', 'Fp2', 'TP10' ]
    
    # EEG freq is 256 Hz according to documentation
    # EEG comes in µV, but 'eeg' takes V
    EEG_info = mne.create_info(ch_names=EEG_ch_names, sfreq=256, ch_types='eeg')
    
    # set metadata                                                                            # 9.5 cm is radius of default model head for topographic map, but
    EEG_info.set_montage(mne.channels.make_standard_montage('standard_1020', head_size=0.09)) # EEG doesn't automatically fit to the circular head so use 0.09
                                                                                              
    # convert to MNE raw type to be processed
    EEG_raw = mne.io.RawArray(data=EEG_data, info=EEG_info)
 
    
    return EEG_raw
    
# Important Note: this function causes the reordering data
def process_fNIRS(fNIRS_raw):

    # convert to optical density and clean
    fNIRS_od = mne.preprocessing.nirs.optical_density(fNIRS_raw)
    fNIRS_od = mne.preprocessing.nirs.temporal_derivative_distribution_repair(fNIRS_od)
    fNIRS_od = mne_nirs.signal_enhancement.short_channel_regression(fNIRS_od, max_dist=0.011) # max_dist set to larger than short channel

    # convert to hemoglobin and clean
    fNIRS_hemo = mne.preprocessing.nirs.beer_lambert_law(fNIRS_od)
    fNIRS_hemo.filter(l_freq=0.01, h_freq=0.1) # range generally reflects speed of blood flow
    
    hemo_spectrum = fNIRS_hemo.compute_psd(fmin=0.01, fmax=0.1)
    
    return fNIRS_hemo, hemo_spectrum

def process_EEG(EEG_raw):
    
    # clean EEG
    EEG_cleaned = EEG_raw.filter(l_freq=1.0, h_freq=40.0)
    
    # convert to PSD
    EEG_spectrum = EEG_cleaned.copy().compute_psd(fmin=1, fmax=40)
    
    return EEG_cleaned, EEG_spectrum

def normalize_EEG(EEG_cleaned):
    EEG_normalized = mne.baseline.rescale(EEG_cleaned.get_data(), EEG_cleaned.times, baseline=(None, None), mode='zscore')
    
    EEG_normalized = mne.io.RawArray(EEG_normalized, EEG_cleaned.info)
    
    EEG_spectrum_normalized = EEG_normalized.copy().compute_psd(fmin=1, fmax=40)
    
    return EEG_normalized, EEG_spectrum_normalized

def separate_EEG_bands(EEG_spectrum):
    
    # get separated bands - these are the same ranges as the compute_psd() function does
    delta = EEG_spectrum.get_data(fmin=0.0, fmax=4.0)
    theta = EEG_spectrum.get_data(fmin=4.0, fmax=8.0)
    alpha = EEG_spectrum.get_data(fmin=8.0, fmax=12.0)
    beta  = EEG_spectrum.get_data(fmin=12.0, fmax=30.0)
    gamma = EEG_spectrum.get_data(fmin=30.0, fmax=45.0)
    
    return delta, theta, alpha, beta, gamma

def separate_average_EEG_bands(EEG_spectrum):
    delta, theta, alpha, beta, gamma = separate_EEG_bands(EEG_spectrum)
    
    # average the specified band to get power value
    delta_avg = delta.mean(axis=1)
    theta_avg = theta.mean(axis=1)
    alpha_avg = alpha.mean(axis=1)
    beta_avg = beta.mean(axis=1)
    gamma_avg = gamma.mean(axis=1)
    
    return delta_avg, theta_avg, alpha_avg, beta_avg, gamma_avg

def export_EEG(EEG_spectrum, EEG_cleaned, filePrefix):
    
    # average EEG scalars
    delta_avg, theta_avg, alpha_avg, beta_avg, gamma_avg = separate_average_EEG_bands(EEG_spectrum)
    
    # make EEG dataframe
    export_EEG_data = {
        'Channel': EEG_cleaned.ch_names,
        'Delta_Mean': delta_avg.tolist(),
        'Theta_Mean':  theta_avg.tolist(),
        'Alpha_Mean':  alpha_avg.tolist(),
        'Beta_Mean':  beta_avg.tolist(),
        'Gamma_Mean':  gamma_avg.tolist()
    }
    
    export_EEG_df = pd.DataFrame(export_EEG_data)
    
    
    # write to csv
    export_EEG_df.to_csv('export/' + filePrefix + 'eeg_processed.csv', index=False)
    
def export_fNIRS(fNIRS_hemo, filePrefix):
    
    # average for fNIRS and EEG scalars
    fNIRS_hemo_avg = fNIRS_hemo.get_data().mean(axis=1)
    
    
    # make fNIRS dataframe
    export_fNIRS_data = {
        'Channel': [ 'left outer HbO', 'right outer HbO', 
                    'left outer HbR', 'right outer HbR',
                    'left inner HbO', 'right inner HbO',
                    'left inner HbR', 'right inner HbR'],
        'Hemo_Mean': fNIRS_hemo_avg.tolist()
    }
    
    export_fNIRS_df = pd.DataFrame(export_fNIRS_data)
    
    # write to csv
    export_fNIRS_df.to_csv('export/' + filePrefix + 'fNIRS_processed.csv', index=False)

def construct_fNIRS_topomap(fNIRS_hemo_avg, fNIRS_hemo):

    hbo_data = fNIRS_hemo_avg[[0, 1]] # long channels 850
    hbr_data = fNIRS_hemo_avg[[2, 3]] # long channels 730
    
    hbo_info = fNIRS_hemo.copy().pick([0, 1]).info
    hbr_info = fNIRS_hemo.copy().pick([2, 3]).info
    
    # topomap calculations break with symmetrical mappings ie div by 0 error,
    # small jitter is added to break up the symmetry
    random = np.random.default_rng(0)
    for i in hbo_info['chs']:
        jitter = random.uniform(-0.0001, 0.0001, 2)
        i['loc'][0] += jitter[0]
        i['loc'][1] += jitter[1]
        
    for i in hbr_info['chs']:
        jitter = random.uniform(-0.0001, 0.0001, 2)
        i['loc'][0] += jitter[0]
        i['loc'][1] += jitter[1]

    fig, (ax1, ax2) = plt.subplots(1, 2)

    mne.viz.plot_topomap(hbo_data, hbo_info, axes=ax1, contours=0, show=False)
    ax1.set_title('Oxyhemoglobin')

    mne.viz.plot_topomap(hbr_data, hbr_info, axes=ax2, contours=0, show=False)
    ax2.set_title('Deoxyhemoglobin')
    
def plot_all(EEG_spectrum, fNIRS_hemo_spectrum, fNIRS_hemo):
    
    # general EEG spectrum graph
    EEG_spectrum.plot()
    plt.show() 
    
    fNIRS_hemo_spectrum.plot()
    plt.show()
    
    # topographic map of EEG
    EEG_spectrum.plot_topomap()
    
    # topographic map of fNIRS
    fNIRS_hemo_avg = fNIRS_hemo.get_data().mean(axis=1)
    construct_fNIRS_topomap(fNIRS_hemo_avg, fNIRS_hemo)
    plt.show()
    
    fNIRS_hemo_avg = fNIRS_hemo.get_data().std(axis=1)
    construct_fNIRS_topomap(fNIRS_hemo_avg, fNIRS_hemo)
    plt.show()
    
    

if __name__ == "__main__":
      
    eeg_files = glob.glob('input/*eeg.csv')
    fNIRS_files = glob.glob('input/*optics.csv')
    
    for file in eeg_files:
    
        ### STEP 1: read data and convert to MNE raw type for processing ###
    
        EEG_df = pd.read_csv(file).dropna()
    
        EEG_raw = convert_EEG_to_raw(EEG_df)
    
    
        ### STEP 2: clean and process data ###
    
        EEG_cleaned, EEG_spectrum = process_EEG(EEG_raw)
    
        EEG_normalized, EEG_spectrum_normalized = normalize_EEG(EEG_cleaned)
    
    
        ### STEP 3: export what i want ###

        prefix = os.path.basename(file).split('eeg.csv')[0]

        export_EEG(EEG_spectrum_normalized, EEG_normalized, prefix)
    
    for file in fNIRS_files:
        
        ### STEP 1: read data and convert to MNE raw type for processing ###
    
        fNIRS_df = pd.read_csv(file).dropna()
    
        fNIRS_raw = convert_fNIRS_to_raw(fNIRS_df)
    
    
        ### STEP 2: clean and process data ###
    
        fNIRS_hemo, hemo_spectrum = process_fNIRS(fNIRS_raw)
    
    
        ### STEP 3: export what i want ###
    
        prefix = os.path.basename(file).split('optics.csv')[0]
        
        export_fNIRS(fNIRS_hemo, prefix)   
    
  
    ### BONUS STEP: visualizations :D ###
    
    # plot_all(EEG_spectrum, hemo_spectrum, fNIRS_hemo)
    
    