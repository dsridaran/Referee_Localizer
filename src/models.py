import os
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
import soundfile as sf
import tensorflow as tf
from scipy.io import wavfile
from scipy.signal import correlate, butter, filtfilt
from itertools import combinations
import random
from feature_extraction import return_files_and_locations, bandpass_filter, get_actual_location

def localize_whistle(file_path, whistle, plot, field_length, field_width, speed_of_sound, grid_size, lowcut, highcut, unique_positions, perturb_threshold = 0, mic_index = None):
    """
    Localize whistle for given set of parameters.

    Parameters:
    file_path (str): File path to snipped audio files.
    whistle (int): The whistle number to localize.
    plot (bool): Returns visualizations if True.
    field_length (float): The length of the field (m).
    field_width (float): The width of the field (m).
    speed_of_sound (float): The speed of sound (m/s).
    grid_size (float): The size of each unit grid over which to localize (m).
    lowcut (float): Low cut frequency for the bandpass filter.
    highcut (float): High cut frequency for the bandpass filter.
    perturb_threshold (float): The limit over which to perturb the position of microphone (m).
    mic_index (float): The index of the microphone to perturb; all microphones are perturbed if None.

    Returns:
    tuple: Contains multiple elements:
        - actual_location (tuple): The actual x, y coordinates of the whistle event.
        - predicted_location (tuple): The predicted x, y coordinates from a single iteration.
        - mean_location_all (tuple): The predicted x, y coordinates from the mean multiple iteration considering all microphone data.
        - mean_location_non_boundary (tuple): The predicted x, y coordinates from the mean multiple iteration (excluding non-boundary points).
        - median_location_all (tuple): The predicted x, y coordinates from the median multiple iteration considering all microphone data.
        - median_location_non_boundary (tuple): The predicted x, y coordinates from the median multiple iteration (excluding non-boundary points).
    """

    # Identify microphone files and coordinates
    mic_files, mic_coordinates = return_files_and_locations(file_path, whistle, perturb_threshold = 0, mic_index = None)
    
    # Create shift matrix   
    shift_matrix, confidence_matrix = create_shift_matrix(mic_files, mic_coordinates, lowcut, highcut)

    # Perform single-iteration approach
    predicted_location, actual_location = find_sound_source(
        whistle = whistle,
        mic_files = mic_files, mic_coordinates = mic_coordinates, 
        shift_matrix = shift_matrix, confidence_matrix = confidence_matrix,
        plot = plot,
        field_length = field_length, field_width = field_width, speed_of_sound = speed_of_sound, grid_size = grid_size,
        unique_positions = unique_positions
    )

    # Perform multi-iteration approach
    mean_location_all, mean_location_non_boundary, median_location_all, median_location_non_boundary = find_sound_source_3(
        whistle,
        mic_files, mic_coordinates, shift_matrix,
        plot,
        field_length = field_length, field_width = field_width, speed_of_sound = speed_of_sound, grid_size = grid_size,
        unique_positions = unique_positions
    )

    if plot:
        print("Actual source location:", actual_location)
        print("Estimated source location (single):", predicted_location)
        print("Estimated source location (mean, all):", mean_location_all)
        print("Estimated source location (mean, non-boundary):", mean_location_non_boundary)
        print("Estimated source location (median, all):", median_location_all)
        print("Estimated source location (median, non_boundary):", median_location_non_boundary)

    return actual_location, predicted_location, mean_location_all, mean_location_non_boundary, median_location_all, median_location_non_boundary

def find_sound_source(whistle, mic_files, mic_coordinates, shift_matrix, confidence_matrix, plot, field_length, field_width, speed_of_sound, grid_size, unique_positions):
    """
    Localize whistle for given set of parameters using single iteration approach.

    Parameters:
    whistle (int): The whistle number to localize.    
    mic_files (list): List of microphone file paths.
    mic_coordinates (list): List of x, y coordinates for microphones.
    shift_matrix (matrix): Optimal shift between pairs of microphones (s).
    confidence_matrix (matrix): Error (post-cross-correlation) for pairs of microphones.
    plot (bool): Returns visualizations if True.
    field_length (float): The length of the field (m).
    field_width (float): The width of the field (m).
    speed_of_sound (float): The speed of sound (m/s).
    grid_size (float): The size of each unit grid over which to localize (m).

    Returns:
    tuple: Contains multiple elements:
        - best_location (tuple): The predicted x, y coordinates from a single iteration.
        - actual_location (tuple): The actual x, y coordinates of the whistle event.
    """
    
    # Create threshold for shift inclusion
    # p75 = np.percentile(confidence_matrix, 75)   
    # threshold_matrix = (confidence_matrix < p75).astype(int)

    # Create grid over the field
    x_grid = np.arange(-field_length / 2, field_length / 2 + grid_size, grid_size)
    y_grid = np.arange(-field_width / 2, field_width / 2 + grid_size, grid_size)

    # Initialize results
    min_error = np.inf
    best_location = None
    error_matrix = np.zeros((len(y_grid), len(x_grid)))

    # Evaluate each grid point
    for xi, x in enumerate(x_grid):
        for yi, y in enumerate(y_grid):

            # Calculate theoretical delays
            theoretical_delays = []
            for coord in mic_coordinates:
                distance = calculate_distance(x, y, coord[0], coord[1])
                theoretical_delays.append(distance / speed_of_sound)

            # Compute errors compared to observed shifts
            error = 0
            for i in range(len(mic_files)):
                for j in range(len(mic_files)):
                    if i != j:
                        observed_delay = shift_matrix[i, j]
                        theoretical_delay = theoretical_delays[i] - theoretical_delays[j]
                        
                        #weight = 1 / confidence_matrix[i, j]
                        #weight = threshold_matrix[i, j]
                        weight = 1
                        
                        error += weight * (observed_delay - theoretical_delay)**2

            # Store error in matrix
            error_matrix[yi, xi] = error

            # Check if this grid point has lowest error
            if error < min_error:
                min_error = error
                best_location = (x, y)

    # Extract actual position
    actual_x, actual_y = get_actual_location(whistle)
    actual_location = (actual_x, actual_y)

    # Plot heatmap
    # plt.figure(figsize = (10, 7))
    # plt.imshow(error_matrix, extent = (x_grid.min(), x_grid.max(), y_grid.min(), y_grid.max()), origin = 'lower', cmap = 'viridis')
    # plt.colorbar(label = 'Error')
    # plt.scatter(*best_location, color = 'red', label = 'Predicted Location')
    # plt.scatter(*actual_location, color = 'green', label = 'Actual Location')
    # plt.xlabel('x (meters)')
    # plt.ylabel('y (meters)')
    # plt.title(f'Sound Source Localization for Whistle {whistle}, Single Iteration')
    # plt.legend()
    # if plot:
    #     plt.show()
    # else:
    #     plt.close()

    return best_location, actual_location

def find_sound_source_3(whistle, mic_files, mic_coordinates, shift_matrix, plot, field_length, field_width, speed_of_sound, grid_size, unique_positions):
    """
    Localize whistle for given set of parameters using single iteration approach.

    Parameters:
    whistle (int): The whistle number to localize.    
    mic_files (list): List of microphone file paths.
    mic_coordinates (list): List of x, y coordinates for microphones.
    shift_matrix (matrix): Optimal shift between pairs of microphones (s).
    plot (bool): Returns visualizations if True.
    field_length (float): The length of the field (m).
    field_width (float): The width of the field (m).
    speed_of_sound (float): The speed of sound (m/s).
    grid_size (float): The size of each unit grid over which to localize (m).

    Returns:
    tuple: Contains multiple elements:
        - all (tuple): The predicted x, y coordinates using mean and all points.
        - non_boundary (tuple): The predicted x, y coordinates using mean and non-boundary points.
        - median_all (tuple): The predicted x, y coordinates using median and all points.
        - median_non_boundary (tuple): The predicted x, y coordinates using median and non-boundary points.
    """

    # Create grid over field
    x_grid = np.arange(-field_length / 2, field_length / 2 + grid_size, grid_size)
    y_grid = np.arange(-field_width / 2, field_width / 2 + grid_size, grid_size)

    # Initialize error matrix
    error_matrix = np.full((len(y_grid), len(x_grid)), np.inf)

    # Initialize counters
    counter_all = 0; running_x_all = 0; running_y_all = 0
    counter_non_boundary = 0; running_x_non_boundary = 0; running_y_non_boundary = 0
    x_pos = []; y_pos = []
    x_pos_non_boundary = []; y_pos_non_boundary = []

    # # Create figure for heatmap
    # plt.figure(figsize = (10, 7))

    # Process each combination of three microphones
    for combo in combinations(range(len(mic_files)), 3):
        min_error = np.inf
        best_location = None

        # Evaluate each grid point
        for xi, x in enumerate(x_grid):
            for yi, y in enumerate(y_grid):

                # Calculate theoretical delays
                theoretical_delays = []
                for idx in combo:
                    coord = mic_coordinates[idx]
                    distance = calculate_distance(x, y, coord[0], coord[1])
                    theoretical_delays.append(distance / speed_of_sound)

                # Compute errors compared to observed shifts
                error = 0
                for i in range(len(combo)):
                    for j in range(len(combo)):
                        if i > j:
                            mic_i = combo[i]
                            mic_j = combo[j]
                            observed_delay = shift_matrix[mic_i, mic_j]
                            theoretical_delay = theoretical_delays[i] - theoretical_delays[j]
                            error += (observed_delay - theoretical_delay)**2

                # Update error matrix with smallest error found
                if error < error_matrix[yi, xi]:
                    error_matrix[yi, xi] = error

                # Check if this grid point has the lowest error for this combination
                if error < min_error:
                    min_error = error
                    best_location = (x, y)

        # Plot predicted location for this combination
        if best_location:

            # # Plot scatter point
            # plt.scatter(*best_location, color = 'gray', s = 10)

            # Track all locations
            counter_all += 1
            running_x_all += best_location[0]
            running_y_all += best_location[1]
            x_pos.append(best_location[0])
            y_pos.append(best_location[1])

            # Track non-boundary locations
            if abs(best_location[0]) < (field_length / 2) and abs(best_location[1]) < (field_width / 2):
                counter_non_boundary += 1
                running_x_non_boundary += best_location[0]
                running_y_non_boundary += best_location[1]
                x_pos_non_boundary.append(best_location[0])
                y_pos_non_boundary.append(best_location[1])

    # Extract actual position
    actual_x, actual_y = get_actual_location(unique_positions, whistle)
    actual_location = (actual_x, actual_y)
    plt.scatter(*actual_location, color = 'green', s = 20, label = 'Actual Location')

    # Extract and plot predictions
    all = (running_x_all / counter_all, running_y_all / counter_all)
    # plt.scatter(*all, color = 'blue', s = 20, label = 'Mean Location (All)')
    
    non_boundary = (running_x_non_boundary / counter_non_boundary, running_y_non_boundary / counter_non_boundary)
    # plt.scatter(*non_boundary, color = 'red', s = 20, label = 'Mean Location (Non-boundary)')
    
    median_all = (np.median(x_pos), np.median(y_pos))
    # plt.scatter(*median_all, color = 'yellow', s = 20, label = 'Median Location (All)')
    
    median_non_boundary = (np.median(x_pos_non_boundary), np.median(y_pos_non_boundary))
    # plt.scatter(*median_non_boundary, color = 'orange', s = 20, label = 'Median Location (Non-boundary)')

    # Plot heatmap
    # plt.imshow(error_matrix, extent = (x_grid.min(), x_grid.max(), y_grid.min(), y_grid.max()), origin = 'lower', cmap = 'viridis', alpha = 0.75)
    # plt.colorbar(label = 'Error')
    # plt.xlabel('X (meters)')
    # plt.ylabel('Y (meters)')
    # plt.title(f'Sound Source Localization for Whistle {whistle}, Multiple Iteration')
    # plt.legend()
    # if plot:
    #     plt.show()
    # else:
    #     plt.close()

    return all, non_boundary, median_all, median_non_boundary

def create_shift_matrix(mic_files, mic_coord, lowcut, highcut):
    """
    Create the time-shift matrix.

    Parameters:
    mic_files (list): List of microphone file paths.
    mic_coordinates (list): List of x, y coordinates for microphones.
    lowcut (float): Low cut frequency for the bandpass filter.
    highcut (float): High cut frequency for the bandpass filter.

    Returns:
    matrix: Optimal shift between pairs of microphones (s).
    matrix: Error (post-cross-correlation) for pairs of microphones.
    """
    
    # Initialize results
    num_mics = len(mic_files)
    shift_matrix = np.zeros((num_mics, num_mics))
    error_matrix = np.zeros((num_mics, num_mics))

    # Iteratively determine optimal shifts and errors
    for i in range(num_mics):
        for j in range(i + 1, num_mics):
            if i != j:
                shift, error = optimal_shift(mic_files[i], mic_files[j], mic_coord[i][0], mic_coord[i][1], mic_coord[j][0], mic_coord[j][1], lowcut, highcut)
                shift_matrix[i, j] = shift
                shift_matrix[j, i] = -shift
                error_matrix[i, j] = abs(error)
                error_matrix[j, i] = abs(error)
    
    return shift_matrix, error_matrix

def optimal_shift(file1, file2, x1, y1, x2, y2, lowcut, highcut, fs = 48000):
    """
    Calculate the optimal shift between two audio signals using cross-correlation.

    Parameters:
    file1 (str): Path to the first audio file.
    file2 (str): Path to the second audio file.
    x1, y1 (float): Coordinates of the microphone for file1.
    x2, y2 (float): Coordinates of the microphone for file2.
    lowcut (float): Low cut frequency for the bandpass filter.
    highcut (float): High cut frequency for the bandpass filter.
    fs (int): Sampling frequency of the audio data.

    Returns:
    float: The optimal shift in seconds between the two signals.
    float: The cross-correlation error (minimum value).
    """

    # Read wave files
    rate1, data1 = wavfile.read(file1)
    rate2, data2 = wavfile.read(file2)

    # Filter sound for desired frequency range
    data1 = bandpass_filter(data1, lowcut, highcut, rate1)
    data2 = bandpass_filter(data2, lowcut, highcut, rate2)

    # Ensure format of data
    if len(data1.shape) > 1:
        data1 = data1[:, 0]
    if len(data2.shape) > 1:
        data2 = data2[:, 0]

    # Define maximum shifts permissible
    d = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    max_time = d / 343
    max_shift = int(round(max_time * fs, 0) + 1)

    # Cross correlate waves
    if max_shift is not None:
        max_shift = min(max_shift, len(data1) - 1, len(data2) - 1)
        data2_padded = np.pad(data2, (max_shift, max_shift), 'constant', constant_values = (0, 0))
        correlation = correlate(data1, data2_padded, mode = 'valid')
    else:
        correlation = correlate(data1, data2, mode = 'full')

    # Determine optimal shift
    min_index = np.argmin(correlation)
    error = correlation[min_index]

    if max_shift is not None:
        shift = (min_index - max_shift) / fs
    else:
        zero_shift_index = len(data1) - 1
        shift = (min_index - zero_shift_index) / fs

    return shift, error

def calculate_distance(x1, y1, x2, y2):
    """
    Calculate distance between two coordinates.

    Parameters:
    x1 (float): x-coordinate of first position (m).
    y1 (float): y-coordinate of first position (m).
    x2 (float): x-coordinate of second position (m).
    y2 (float): y-coordinate of second position (m).

    Returns:
    float: The distance between the two coordinates (m).
    """
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)