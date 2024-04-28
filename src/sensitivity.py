import src
from models import localize_whistle
from data_preprocessing import load_unique_ref_positions

def iterate_all_mics(match, file_path = "../data/snipped", whistles = range(2, 161), field_length = 120, field_width = 80, speed_of_sound = 343, grid_size = [10, 1], lowcut = 3750, highcut = 4250, perturb_threshold = 0, mic_index = None, seeds = [123], half_1 = 999999999, half_2 = 999999999, half_3 = 999999999, half_4 = 999999999):
    """
    Localize all microphones across a match for a given set of input parameters.

    Parameters:
    match (str): String identifying match to be processed.
    file_path (str): File path to snipped audio files.
    whistles (list): The whistle numbers to localize.
    field_length (float): The length of the field (m).
    field_width (float): The width of the field (m).
    speed_of_sound (float): The speed of sound (m/s).
    grid_size (list): The size(s) of each unit grid over which to localize (m).
    lowcut (float): Low cut frequency for the bandpass filter.
    highcut (float): High cut frequency for the bandpass filter.
    perturb_threshold (float): The distance limit over which to perturb the position of microphone (m).
    mic_index (float): The index of the microphone to perturb; all microphones are perturbed if None.
    seeds (list): List of seeds to apply to localization.
    comb_size (float): Number of microphones (in each combination) to use in localization.
    half_1 (int): Starting sample (based on 48kHz) of first half.
    half_2 (int): Starting sample (based on 48kHz) of second half.
    half_3 (int): Starting sample (based on 48kHz) of first half of extra time (if required).
    half_4 (int): Starting sample (based on 48kHz) of second half of extra time (if required).

    Returns:
    None
    """
    
    # Set appropriate file path for match
    file_path = f'{file_path}/{match}'

    # Load referee's actual positions (for validation)
    unique_positions = load_unique_ref_positions(
        f"../data/meta/{match}/whistles_{match}_Tagged.csv", 
        f"../data/meta/{match}/refs_track_{match}.parquet",
        half_1, half_2, half_3, half_4
    )
    
    # Initialize list to store results
    results = []

    # Identify number of microphones used
    if mic_index is None:
        num_mics = 18
    else:
        num_mics = len(mic_index)

    # Iterate all whistles and seeds
    for whistle_id in tqdm(whistles, "Estimated Completion (By Whistle)"):
        for seed in seeds:
            try:
                
                # Localize referee
                start_time = time.time()
                actual, pred_mean, pred_median = localize_whistle(
                    file_path = file_path, whistle = whistle_id, plot = False,
                    field_length = field_length, field_width = field_width, speed_of_sound = speed_of_sound, grid_size = grid_size,
                    lowcut = lowcut, highcut = highcut, 
                    unique_positions = unique_positions,
                    perturb_threshold = perturb_threshold, mic_index = mic_index,
                    seed = seed
                )
                end_time = time.time()
        
                # Append results
                results.append({
                    "event_id": whistle_id,
                    "run_time": end_time - start_time,
                    "seed": seed,
                    "field_length": field_length,
                    "field_width": field_width,
                    "speed_of_sound": speed_of_sound,
                    "grid_size": grid_size,
                    "num_mics": num_mics,
                    "perturb_threshold": perturb_threshold,
                    "mic_index": mic_index,
                    "lowcut": lowcut,
                    "highcut": highcut,
                    "x_actual": actual[0],
                    "y_actual": actual[1],
                    "x_mean": pred_mean[0],
                    "y_mean": pred_mean[1],
                    "x_median": pred_median[0],
                    "y_median": pred_median[1]
                })
                
            except Exception as e:
                continue

    # Calculate errors
    results_df = pd.DataFrame(results)
    results_df['error_mean'] = ((results_df['x_actual'] - results_df['x_mean'])**2 + (results_df['y_actual'] - results_df['y_mean'])**2)**0.5
    results_df['error_median'] = ((results_df['x_actual'] - results_df['x_median'])**2 + (results_df['y_actual'] - results_df['y_median'])**2)**0.5

    # Create output folder
    folder_path = f'../data/results/{match}'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    # Save file
    if perturb_threshold == 0:
        file_name = f'{match}_l{field_length}_w{field_width}_g{grid_size}_s{speed_of_sound}_lc{lowcut}_hc{highcut}_n{num_mics}_p{perturb_threshold}_m{mic_index}'
    else:
        file_name = f'{match}_e{whistles}_l{field_length}_w{field_width}_g{grid_size}_s{speed_of_sound}_lc{lowcut}_hc{highcut}_n{num_mics}_p{perturb_threshold}_m{mic_index}'
    
    results_df.to_csv(f'../data/results/{match}/{file_name}.csv', index = False)