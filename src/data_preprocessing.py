import os
import numpy as np
import pandas as pd
import librosa
import soundfile as sf

def load_tagged_whistles(filepath, half_1 = 8107245, half_2 = 188327149, half_3 = 344150253, half_4 = 405575917):
    """
    Load tagged whistle data from a CSV file.

    Parameters:
    filepath (str): The path to the CSV file.

    Returns:
    DataFrame: A pandas DataFrame containing the tagged whistle data.
    """
    whistles_tagged = pd.read_csv(filepath)
    whistles_filtered = whistles_tagged[whistles_tagged['Tag'] == 1].reset_index(drop = True)
    whistles = create_half_and_time_columns(whistles_filtered, half_1, half_2, half_3, half_4)
    return whistles

def round_to_nearest_40(x):
    return max(40, round(x / 40) * 40)

def load_unique_ref_positions(whistles_path, ref_track_path):
    whistles = load_tagged_whistles(whistles_path)
    ref_track = pd.read_parquet(ref_track_path)
    master_df = pd.merge(whistles, ref_track, on = ['time_ms', 'half'], how = 'left')
    unique_positions = master_df.drop_duplicates(subset = 'event_id', keep = 'first')[['event_id', 'x', 'y']]
    unique_positions = unique_positions.reset_index(drop = True)
    unique_positions = unique_positions.dropna()
    return unique_positions

def create_half_and_time_columns(whistles, half_1, half_2, half_3, half_4):
    """
    Assign half based on sample boundaries and determine time of whistle (in ms).

    Parameters:
    whistles (DataFrame): DataFrame with whistle data including 'Sample' column.
    boundaries (list): List of sample numbers that define the boundaries of the halves.

    Returns:
    DataFrame: The input DataFrame with a new column 'half' and 'time_ms'.
    """

    # Identify relevant half for match
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

    # Identify whistle events
    event_id = 1
    event_start_time = whistles.iloc[0]['Sample']
    whistles['event_id'] = None
    whistles.at[0, 'event_id'] = event_id
    
    for i in range(1, len(whistles)):
        if whistles.iloc[i]['Sample'] - event_start_time <= 18500:
            whistles.at[i, 'event_id'] = event_id
        else:
            event_id += 1
            whistles.at[i, 'event_id'] = event_id
            event_start_time = whistles.iloc[i]['Sample']
    
    return whistles

def recut_data(recut_flag, whistles, data_folder = "../data", padding = 96000):
    """
    Recut sound data (if required).

    Parameters:
    recut_flag (bool): Re-cuts sound clips if True.
    whistles (DataFrame): DataFrame containing timestamped whistles.
    data_folder (str): Folder path to raw data.
    padding (int): Number of samples to include before and after the whistle event.

    Returns:
    None
    """
    
    if recut_flag:      
        files = os.listdir(f'{data_folder}/raw')
        filenames_without_extension = [os.path.splitext(file)[0] for file in files if os.path.isfile(os.path.join(f'{data_folder}/raw', file))]
        for file in filenames_without_extension:
            print(f'Splitting file: {file}')
            split_sound_data(whistles, file, data_folder, padding) 

def split_sound_data(whistles, file, file_path, padding):
    """
    Split sound data from a given file and save snippets based on events.

    Parameters:
    file (str): The filename of the audio file.
    file_path (str): The path to the data.
    padding (int): Number of samples to include before and after the whistle event.
    whistles (DataFrame): DataFrame containing timestamped whistles.

    Returns:
    None
    """
   
    # Load audio file
    audio_path = f'{file_path}/raw/{file}.wav'
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
        output_filename = f'{file_path}/snipped/{event_id}_{file}.wav'
        sf.write(output_filename, trimmed_audio, sr)