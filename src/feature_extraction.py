import numpy as np
from scipy.signal import butter, filtfilt, correlate
from scipy.io import wavfile

def return_files_and_locations(file_path, whistle, perturb_threshold = 0, mic_index = None):
    """
    Return list of sound files and corresponding locations.

    Parameters:
    file_path (str): File path to snipped audio files.
    whistle (int): Number of whistle to be localized.
    perturb_threshold (float): The limit over which to perturb the position of microphone (m).
    mic_index (float): The index of the microphone to perturb; all microphones are perturbed if None.

    Returns:
    tuple: Contains multiple elements:
        - mic_files (list): List of microphone file paths.
        - mic_coordinates (list): List of x, y coordinates for microphones.
    """
    
    # Define sound files
    mic_files = [
        f'{file_path}/{whistle}_104_NEAR_LEFT_CORNER.wav',
        f'{file_path}/{whistle}_105_LEFT_GOAL_-_NEAR.wav',
        f'{file_path}/{whistle}_106_LEFT_GOAL_-_CAMERA.wav',
        f'{file_path}/{whistle}_107_LEFT_GOAL_-_FAR.wav',
        f'{file_path}/{whistle}_108_FAR_LEFT_CORNER.wav',
        f'{file_path}/{whistle}_109_FAR_LEFT_18YRD.wav',
        f'{file_path}/{whistle}_110_FAR_CENTRE_-_LEFT.wav',
        f'{file_path}/{whistle}_111_FAR_CENTRE_-_CAMERA.wav',
        f'{file_path}/{whistle}_112_FAR_CENTRE_-_RIGHT.wav',
        f'{file_path}/{whistle}_113_FAR_RIGHT_18YRD.wav',
        f'{file_path}/{whistle}_114_FAR_RIGHT_CORNER.wav',
        f'{file_path}/{whistle}_115_RIGHT_GOAL_-_FAR.wav',
        f'{file_path}/{whistle}_116_RIGHT_GOAL_-_CAMERA.wav',
        f'{file_path}/{whistle}_117_RIGHT_GOAL_-_NEAR.wav',
        f'{file_path}/{whistle}_118_NEAR_RIGHT_CORNER.wav',
        f'{file_path}/{whistle}_119_NEAR_RIGHT_18YRD.wav',
        f'{file_path}/{whistle}_120_NEAR_CENTRE_-_CAMERA_3.wav',
        f'{file_path}/{whistle}_121_NEAR_LEFT_18YRD.wav'
    ]

    # Define microphone coordinates
    mic_coordinates = [
        (-54.700, -36.025), (-55.575, -3.950), (-59.950, -10.900), (-55.700, 3.950), (-54.775,  36.050), (-34.225, 36.275), 
        (-9.050, 36.250), (0, 37.250), (9.050, 36.300), (34.200, 36.325), (54.500, 36.225), (55.425, 3.950), 
        (60.050, -10.70), (55.360, -3.900), (54.675, -36.200), (34.225, -36.350), (0, -36.875), (-34.375, -37.025)
    ]

    # Perturb microphone coordinates
    def perturb_coordinate(coord):
        return (coord[0] + random.uniform(-perturb_threshold, perturb_threshold), coord[1] + random.uniform(-perturb_threshold, perturb_threshold))

    if perturb_threshold > 0:
        if mic_index is not None:
            mic_coordinates[mic_index] = perturb_coordinate(mic_coordinates[mic_index])
        else:
            mic_coordinates = [perturb_coordinate(coord) for coord in mic_coordinates]

    return mic_files, mic_coordinates

def bandpass_filter(data, lowcut, highcut, fs, order = 5):
    """
    Apply a band-pass filter to the data array.

    Parameters:
    data (array): The audio data to filter.
    lowcut (float): The low cut frequency in Hz.
    highcut (float): The high cut frequency in Hz.
    fs (int): The sampling frequency of the data.
    order (int): The order of the filter.

    Returns:
    array: The filtered data.
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype = 'band')
    filtered_data = filtfilt(b, a, data)
    return filtered_data

def get_actual_location(unique_positions, whistle):
    """
    Determine actual referee position for whistle.

    Parameters:
    unique_positions (DataFrame): DataFrame of unique positions for all whistles.
    whistle (int): Number of whistle to be localized.

    Returns:
    actual_location (tuple): The actual x, y coordinates of the whistle event.
    """
    filtered_df = unique_positions[unique_positions['event_id'] == whistle]
    return filtered_df['x'].iloc[0], filtered_df['y'].iloc[0]