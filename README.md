![Logo](/images/logo.png "Logo")

# Referee Localizer

## Description
This project provides a system for performing referee triangulation using sound data captured from multiple microphones placed around a sports field. The goal is to accurately localize the position of referees based on whistle sounds during matches. This system is valuable for enhancing the accuracy of referee tracking technologies.

![Referee Localization Example](/images/gif.gif "Localization Diagram")

## Installation
To set up this project, clone the repository to your local machine and ensure that you have Jupyter Notebook installed to run `.ipynb` files.

```bash
git clone https://github.com/dsridaran/Referee_Localizer.git
cd Referee_Localizer
```

## Usage

### Preprocessing Data

Before performing localization, preprocess the input data by running the following Jupyter notebook:

```bash
jupyter notebook preprocess_whistles_data.ipynb
```

This script will preprocess the raw audio data from 18 microphones, tag the whistles, and prepare the data for localization.

### Performing Localization

Proceed to localization using the following Jupyter notebook:

```bash
jupyter notebook perform_localization.ipynb
```

**Example: Localizing a Single Whistle**

To localize a single whistle, use the following snippet in the perform_localization.ipynb notebook:

```bash
# Load known referee positions (for validation)
unique_positions = load_unique_ref_positions(
    f"../data/meta/BRA_CRO/whistles_XXX_Tagged.csv",
    f"../data/meta/BRA_CRO/refs_track_XXX.parquet",
    half_1 = XXX, half_2 = XXX, half_3 = XXX, half_4 = XXX
)

# Localize single whistle
actual, pred_mean, pred_median = localize_whistle(
    file_path = "../data/snipped/XXX",
    whistle = XXX,
    plot = True
)
```

**Example: Localizing All Whistles in a Match**

To process all whistles for a specific match, use the following snippet in the perform_localization.ipynb notebook:

```bash
# Localize all whistles using baseline parameters
iterate_all_mics(match = XXX, half_1 = XXX, half_2 = XXX, half_3 = XXX, half_4 = XXX, whistles = XXX)
```

## Input Data Structure

The expected inputs for each match are organized as follows:

- **data/raw/{match}/{microphone}.wav:** 48kHz audio data from 18 microphones. This data was provided by Salsa Sound.
- **data/meta/{match}/whistles_{match}_Tagged.csv:** Human-approximated tagged whistle times. This data was provided by Salsa Sound.
- **data/meta/{match}/microphone_positions.csv:** Positional coordinates of each microphone. This data was provided by Salsa Sound.
- **data/meta/{match}/refs_track_{match}.parquet:** 40Hz referee tracking data for validation. This data was provided by FIFA.

## Contact

For questions or support on this code, please contact Dilan SriDaran (dilan.sridaran@gmail.com).

For requests to access the Salsa data, please contact Rob Oldfield (rob@salsasound.com).

For requests to access the FIFA data, please contact Johsan Billingham (johsan.billingham@fifa.org).

If you are interested in contributing to this research, please contact Johsan Billingham (johsan.billingham@fifa.org).
