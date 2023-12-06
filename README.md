# Cooling for Comfort, Warming the World
## Residential and Office Cooling and its Environmental Implications in The Hague

### Introduction

This repository contains the codebase for a MSc thesis in Industrial Ecology on the topic of residential and office cooling and its environmental impacts in the city of The Hague, The Netherlands. It was authored by Simon van Lierde under the guidance of Dr. Ir. Peter Luscuere and Dr. Benjamin Sprecher.

### Directory & File Structure

1. **Functions**

    This directory contains a collection of Python scripts that provide various functionalities required for the project. Here's a breakdown of the scripts:

    - **data_handling.py**: Contains tha main functions related to data processing and handling.
    - **environmental.py**: Functions related to environmental calculations and metrics.
    - **figures.py**: Functions to generate and handle figures for visualization of results.
    - **geometric.py**: Geometric calculations and related functions.
    - **sensitivity_analysis.py**: Functions for performing sensitivity analysis.
    - **thermodynamic.py**: Thermodynamic calculations and related functions.
    - **time_series.py**: Generation and processing of weather data and other time series.

2. **Notebooks**

    - **gis.ipynb**: A Jupyter notebook focused on preparing the GIS (Geographical Information System) data from the BAG (*Basisregistratie Adressen en Gebouwen*; Registry of Addresses and Buildings), related tasks and spatial visualizations.
    - **main.ipynb**: The main Jupyter notebook that integrates the calculation of the cooling demand for buildings in The Hague and related environmental impacts, as well as sensitivity analyses and visualization of results.

3. **Data**
    - ***input/parameters/***: Contains all input parameters used in the thermodynamic and environmental impact modeling. Larger spatial datasets used in the model are hosted separately at https://zenodo.org/doi/10.5281/zenodo.8344580.
    - ***output/***: Contains the main model results, aggregated by building type and energy label.

4. **Project Configuration**

    - **requirements.txt**: Lists all the Python packages required to run the project.

## Getting Started

1. Clone the repository to your local machine.
2. Install the required Python packages using the command:

    ```
    pip install -r requirements.txt
    ```
3. Download the required spatial datasets used as input from Zenodo: https://zenodo.org/doi/10.5281/zenodo.8344580
4. Launch Jupyter Notebook to access the `gis.ipynb` and `main.ipynb` notebooks.

## Contribution

Feel free to fork the repository, make changes, and submit pull requests. Any contributions to improve the project are welcome.

## License

Please refer to the repository's license file for usage and distribution guidelines.