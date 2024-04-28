import src

def localize_whistle(file_path, whistle, plot, field_length, field_width, speed_of_sound, grid_size, lowcut, highcut, unique_positions, seed, perturb_threshold = 0, mic_index = None):
    """
    Localize whistle for given set of parameters.

    Parameters:
    file_path (str): File path to snipped audio files.
    whistle (int): The whistle number to localize.
    plot (bool): Returns visualizations if True.
    field_length (float): The length of the field (m).
    field_width (float): The width of the field (m).
    speed_of_sound (float): The speed of sound (m/s).
    grid_size (float): The size(s) of each unit grid over which to localize (m).
    lowcut (float): Low cut frequency for the bandpass filter.
    highcut (float): High cut frequency for the bandpass filter.
    unique_positions (DataFrame): DataFrame of referee's actual location (for validation).
    seed (int): Random seed for sensitivity analysis.
    perturb_threshold (float): The distance limit over which to perturb the position of microphone (m).
    mic_index (float): The index of the microphone to perturb; all microphones are perturbed if None.

    Returns:
    tuple: Contains multiple elements:
        - actual (tuple): The actual x, y coordinates of the whistle event.
        - pred_mean (tuple): The predicted x, y coordinates from the mean multiple iteration (excluding non-boundary points).
        - pred_median (tuple): The predicted x, y coordinates from the median multiple iteration (excluding non-boundary points).
    """

    # Extract relevant microphone files and coordinates
    mic_files, mic_coordinates = return_files_and_locations(file_path, whistle, seed, perturb_threshold = perturb_threshold, mic_index = mic_index)
    
    # Create time shift matrix   
    shift_matrix = create_shift_matrix(mic_files, mic_coordinates, lowcut, highcut)

    # Initial iterate through grid sizes
    start_x = 0
    start_y = 0
    
    for g in grid_size:
    
        # Localize whistle based on grid size and field dimensions
        actual, pred_mean, pred_median = find_sound_source(
            whistle = whistle,
            mic_files = mic_files, mic_coordinates = mic_coordinates, shift_matrix = shift_matrix,
            plot = plot,
            field_length = field_length, field_width = field_width, 
            speed_of_sound = speed_of_sound, grid_size = g,
            unique_positions = unique_positions,
            init_x = start_x, init_y = start_y
        )
        
        # Print results
        if plot:
            print(f"Starting point: ({start_x}, {start_y}), with grid size: {g}")
            print("Actual source location:", actual)
            print("Predicted source location (mean):", pred_mean)
            print("Predicted source location (median):", pred_median)
            
        # Update starting guess for next iteration
        start_x = pred_median[0]
        start_y = pred_median[1]

        # Update field size for next iteration
        field_length = 3 * g
        field_width = 3 * g 
   
    return actual, pred_mean, pred_median
    
def find_sound_source(whistle, mic_files, mic_coordinates, shift_matrix, plot, field_length, field_width, speed_of_sound, grid_size, unique_positions, init_x = 0, init_y = 0):
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
    unique_positions (DataFrame): DataFrame of referee's actual location (for validation).
    init_x (float): Centre of grid (x-coordinate) over which to search (m).
    init_y (float): Centre of grid (y-coordinate) over which to search (m).

    Returns:
    tuple: Contains multiple elements:
        - all (tuple): The predicted x, y coordinates using mean and all points.
        - non_boundary (tuple): The predicted x, y coordinates using mean and non-boundary points.
        - median_all (tuple): The predicted x, y coordinates using median and all points.
        - median_non_boundary (tuple): The predicted x, y coordinates using median and non-boundary points.
    """

    # Create grid (centered around initial location)
    x_grid = np.arange(init_x - field_length / 2, init_x + field_length / 2 + grid_size, grid_size)
    y_grid = np.arange(init_y - field_width / 2, init_y + field_width / 2 + grid_size, grid_size)
    x_mesh, y_mesh = np.meshgrid(x_grid, y_grid)
    
    # Compute theoretical time delays for each grid tile
    all_delays = np.zeros((len(mic_coordinates), len(y_grid), len(x_grid)))
    for idx, coord in enumerate(mic_coordinates):
        distances = np.sqrt((x_mesh - coord[0])**2 + (y_mesh - coord[1])**2)
        all_delays[idx] = distances / speed_of_sound

    # Initialize results
    error_matrix = np.full((len(y_grid), len(x_grid)), np.inf)
    counter = 0; running_x = 0; running_y = 0; x_pos = []; y_pos = []
    if plot:
        plt.figure(figsize = (10, 7))

    # Process each combination of three microphones
    for combo in combinations(range(len(mic_files)), 3):
        
        # Initialize results for microphone combination
        min_error = np.inf
        best_location = None
        
        # Extract theoretical delays
        delays = all_delays[list(combo)]

        # Evaluate each grid point
        for xi, x in enumerate(x_grid):
            for yi, y in enumerate(y_grid):

                # Compute errors compared to observed shifts
                error = 0
                for i in range(len(combo)):
                    for j in range(len(combo)):
                        if i > j:
                            mic_i = combo[i]
                            mic_j = combo[j]
                            observed_delay = shift_matrix[mic_i, mic_j]
                            theoretical_delay = delays[i, yi, xi] - delays[j, yi, xi]
                            error += (observed_delay - theoretical_delay)**2

                # Update error matrix with smallest error found
                if error < error_matrix[yi, xi]:
                    error_matrix[yi, xi] = error

                # Update prediction if this grid point has the lowest error for this microphone combination
                if error < min_error:
                    min_error = error
                    best_location = (x, y)

        # Extract best predictions within field
        if best_location:
            if (init_x - field_length / 2) < best_location[0] < (init_x + field_length / 2) and (init_y - field_width / 2) < best_location[1] < (init_y + field_width / 2):
                plt.scatter(*best_location, color = 'gray', s = 10)
                counter += 1
                running_x += best_location[0]
                running_y += best_location[1]
                x_pos.append(best_location[0])
                y_pos.append(best_location[1])

    # Extract actual locations
    actual_x, actual_y = get_actual_location(unique_positions, whistle)
    actual = (actual_x, actual_y)

    # Extract predicted locations
    if counter > 0:
        pred_mean = (running_x / counter, running_y / counter)
        pred_median = (np.median(x_pos), np.median(y_pos))
    else:
        pred_mean = pred_median = (0, 0)

    # Plot heatmap (if required)
    if plot:
        plt.scatter(*actual, color = 'green', s = 20, label = 'Actual Location')
        plt.scatter(*pred_median, color = 'blue', s = 20, label = 'Predicted Location')
        plt.imshow(error_matrix, extent = (x_grid.min(), x_grid.max(), y_grid.min(), y_grid.max()), origin = 'lower', cmap = 'viridis', alpha = 0.75)
        plt.colorbar(label = 'Error')
        plt.xlabel('X (meters)')
        plt.ylabel('Y (meters)')
        plt.title(f'Whistle {whistle} Localization: Grize Size {grid_size}')
        plt.legend()
        plt.show()

    return actual, pred_mean, pred_median

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
    """
    
    # Load all microphone files and apply bandpass filter
    audio_data = {}
    for file in mic_files:
        rate, data = wavfile.read(file)
        data = bandpass_filter(data, lowcut, highcut, rate)
        if len(data.shape) > 1:
            data = data[:, 0]
        audio_data[file] = data
    
    # Initialize results
    num_mics = len(mic_files)
    shift_matrix = np.zeros((num_mics, num_mics))

    # Iteratively determine optimal shifts for pairs of microphones
    for i in range(num_mics):
        for j in range(i + 1, num_mics):
            if i != j:
                shift = optimal_shift(audio_data[mic_files[i]], audio_data[mic_files[j]], mic_coord[i][0], mic_coord[i][1], mic_coord[j][0], mic_coord[j][1])
                shift_matrix[i, j] = shift
                shift_matrix[j, i] = -shift
    
    return shift_matrix
    
def optimal_shift(data1, data2, x1, y1, x2, y2, fs = 48000):
    """
    Calculate optimal shift between two audio signals using cross-correlation.

    Parameters:
    data1 (str): Audio file for first microphone.
    data2 (str): Audio file for second microphone.
    x1, y1 (float): Coordinates of the microphone for data1.
    x2, y2 (float): Coordinates of the microphone for data2.
    fs (int): Sampling frequency of the audio data.

    Returns:
    float: The optimal shift in seconds between the two signals.
    """

    # Define maximum shifts permissible based on Euclidean distances
    d = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    max_time = d / 343
    max_shift = int(round(max_time * fs, 0) + 1)
    max_shift = min(max_shift, len(data1) - 1, len(data2) - 1)

    # Cross correlate waves
    data2_padded = np.pad(data2, (max_shift, max_shift), 'constant', constant_values = (0, 0))
    correlation = correlate(data1, data2_padded, mode = 'valid')

    # Extract optimal shift
    min_index = np.argmin(correlation)
    shift = (min_index - max_shift) / fs

    return shift

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

def return_files_and_locations(file_path, whistle, seed, perturb_threshold = 0, mic_index = None):
    """
    Return list of sound files and corresponding locations.

    Parameters:
    file_path (str): File path to snipped audio files.
    whistle (int): Number of whistle to be localized.
    seed (int): Seed for reproducability of positional pertubations.
    perturb_threshold (float): The limit over which to perturb the position of microphone (m).
    mic_index (float): The index of the microphone to perturb; all microphones are perturbed if None.

    Returns:
    tuple: Contains multiple elements:
        - mic_files (list): List of microphone file paths.
        - mic_coordinates (list): List of x, y coordinates for microphones.
    """
    
    # Define location of all sound files
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

    # Define coordinates of all microphones
    mic_coordinates = [
        (-54.7, -36.025), 
        (-55.575, -3.95), 
        (-59.95, -10.9), 
        (-55.7, 3.95), 
        (-54.775, 36.05), 
        (-34.225, 36.275), 
        (-9.05, 36.25), 
        (0, 37.25), 
        (9.05, 36.3), 
        (34.2, 36.325), 
        (54.5, 36.225), 
        (55.425, 3.95), 
        (60.05, -10.7), 
        (55.36, -3.9), 
        (54.675, -36.2), 
        (34.225, -36.35), 
        (0, -36.875), 
        (-34.375, -37.025)
    ]
    
    # Select sample of microphones (if required)
    if mic_index is not None:
        mic_coordinates = [mic_coordinates[i] for i in mic_index]
        mic_files = [mic_files[i] for i in mic_index]

    # Randomly perturb microphone coordinates (if required)
    def perturb_coordinate(coord):
        return (coord[0] + random.uniform(-perturb_threshold, perturb_threshold), coord[1] + random.uniform(-perturb_threshold, perturb_threshold))

    if perturb_threshold > 0:
        random.seed(seed)
        np.random.seed(seed)
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