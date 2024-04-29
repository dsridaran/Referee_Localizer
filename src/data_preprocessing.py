import os
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
import soundfile as sf
from scipy.io import wavfile
from scipy.signal import correlate, butter, filtfilt
import tensorflow as tf
from itertools import combinations
from tqdm import tqdm
import time
import random

def load_tagged_whistles(filepath, half_1 = 999999999, half_2 = 999999999, half_3 = 999999999, half_4 = 999999999):
    """
    Load tagged whistle data from a CSV file.

    Parameters:
    filepath (str): The path to the tagged whistles CSV file.
    half_1 (int): Starting sample (based on 48kHz) of first half.
    half_2 (int): Starting sample (based on 48kHz) of second half.
    half_3 (int): Starting sample (based on 48kHz) of first half of extra time (if required).
    half_4 (int): Starting sample (based on 48kHz) of second half of extra time (if required).

    Returns:
    DataFrame: A pandas DataFrame containing the tagged whistle data.
    """
    
    # Load CSV file
    whistles_tagged = pd.read_csv(filepath)
    
    # Filter only tagged (or identified) whistles
    whistles_filtered = whistles_tagged[whistles_tagged['Tag'] == 1].reset_index(drop = True)
    
    # Create features for match half and time
    whistles = create_half_and_time_columns(whistles_filtered, half_1, half_2, half_3, half_4)
    return whistles

def create_half_and_time_columns(whistles, half_1, half_2, half_3, half_4, cutoff = 18500):
    """
    Assign half and determine time of whistle (in ms) based on sample boundaries.

    Parameters:
    whistles (DataFrame): DataFrame with whistle data including 'Sample' column.
    half_1 (int): Starting sample (based on 48kHz) of first half.
    half_2 (int): Starting sample (based on 48kHz) of second half.
    half_3 (int): Starting sample (based on 48kHz) of first half of extra time (if required).
    half_4 (int): Starting sample (based on 48kHz) of second half of extra time (if required).
    cutoff (int): Time difference threshold (in 48kHz samples) for two whistles to be idnetified as unique events.

    Returns:
    DataFrame: The input DataFrame with a new column 'half' and 'time_ms'.
    """

    # Identify relevant half of whistle based on half thresholds
    conditions = [
        (whistles['Sample'] < half_2),
        (whistles['Sample'] >= half_2) & (whistles['Sample'] < half_3),
        (whistles['Sample'] >= half_3) & (whistles['Sample'] < half_4),
        (whistles['Sample'] >= half_4)
    ]
    choices = [1, 2, 3, 4]
    whistles['half'] = np.select(conditions, choices, default = 5)
    
    # Create time of whistle in ms
    choices = [half_1, half_2, half_3, half_4]
    whistles['half_start'] = np.select(conditions, choices, default = 0)
    whistles['time_ms'] = (whistles['Sample']  - whistles['half_start']) / 48
    whistles['time_ms'] = whistles['time_ms'].apply(round_to_nearest_40)

    # Identify unique whistle events
    event_id = 1
    event_start_time = whistles.iloc[0]['Sample']
    whistles['event_id'] = None
    whistles.at[0, 'event_id'] = event_id
    
    for i in range(1, len(whistles)):
        # Assume whistle belongs to same event if delay is less than cutoff
        if whistles.iloc[i]['Sample'] - event_start_time <= cutoff:
            whistles.at[i, 'event_id'] = event_id
        else:
        # Create new whistle event if delay exceeds cutoff
            event_id += 1
            whistles.at[i, 'event_id'] = event_id
            event_start_time = whistles.iloc[i]['Sample']
    
    return whistles

def round_to_nearest_40(x):
    """
    Rounds number to nearest 40.

    Parameters:
    x (float): Number.

    Returns:
    float: The input number rounded to the nearest 40.
    """
    return max(40, round(x / 40) * 40)

def load_unique_ref_positions(whistles_path, ref_track_path, half_1 = 999999999, half_2 = 999999999, half_3 = 999999999, half_4 = 999999999):
    """
    Identifies actual location of referee at whistle events using TRACAB data.

    Parameters:
    whistles_path (str): The path to the tagged whistles CSV file.
    ref_track_path (str): The path to the referee tracking data.
    half_1 (int): Starting sample (based on 48kHz) of first half.
    half_2 (int): Starting sample (based on 48kHz) of second half.
    half_3 (int): Starting sample (based on 48kHz) of first half of extra time (if required).
    half_4 (int): Starting sample (based on 48kHz) of second half of extra time (if required).

    Returns:
    DataFrame: A pandas DataFrame containing the actual positional coordinates of the referee for each whistle.
    """
    
    # Load relevant files
    whistles = load_tagged_whistles(whistles_path, half_1, half_2, half_3, half_4)
    ref_track = pd.read_parquet(ref_track_path)
    
    # Merge files based on time_ms and half
    master_df = pd.merge(whistles, ref_track, on = ['time_ms', 'half'], how = 'left')
    
    # Remove duplicates to isolate unique events
    unique_positions = master_df.drop_duplicates(subset = 'event_id', keep = 'first')[['event_id', 'x', 'y']]
    unique_positions = unique_positions.reset_index(drop = True)
    unique_positions = unique_positions.dropna()
    return unique_positions

def recut_data(recut_flag, whistles, match, data_folder = "../data", padding = 96000):
    """
    Cut full audio files into snippets required for models.

    Parameters:
    recut_flag (bool): Re-cuts sound clips if True.
    whistles (DataFrame): DataFrame containing timestamped whistles.
    match (str): String identifying match to be processed.
    data_folder (str): Folder path to raw data.
    padding (int): Number of samples to include before and after the whistle event.

    Returns:
    None
    """
    
    # Process data if required
    if recut_flag:  
        
        # Identify all sound files
        files = os.listdir(f'{data_folder}/raw/{match}')
        filenames_without_extension = [os.path.splitext(file)[0] for file in files if os.path.isfile(os.path.join(f'{data_folder}/raw/{match}', file))]
        
        # Iteratively cut sound files
        for file in filenames_without_extension:
            print(f'Splitting file: {file}')
            split_sound_data(whistles, file, data_folder, match, padding) 

def split_sound_data(whistles, file, file_path, match, padding):
    """
    Split full match audio from a given microphone into audio around whistle events.

    Parameters:
    whistles (DataFrame): DataFrame containing timestamped whistles.
    file (str): The filename of the audio file.
    file_path (str): The path to the data.
    match (str): String identifying match to be processed.
    padding (int): Number of samples to include before and after the whistle event.
    
    Returns:
    None
    """
   
    # Load full audio file
    audio_path = f'{file_path}/raw/{match}/{file}.wav'
    audio, sr = librosa.load(audio_path, sr = None)
    
    # Identify starting frame for each whistles
    starting_time = whistles.groupby('event_id')['Sample'].min().reset_index()

    # Iteratively split sound file
    for index, row in starting_time.iterrows():

        # Calculate start and end frame for event
        event_id = row['event_id']
        sample_frame = row['Sample']
        start_frame = max(sample_frame - padding, 0)
        end_frame = min(sample_frame + padding, len(audio))

        # Trim audio
        trimmed_audio = audio[start_frame:end_frame]
        output_filename = f'{file_path}/snipped/{match}/{event_id}_{file}.wav'
        sf.write(output_filename, trimmed_audio, sr)
