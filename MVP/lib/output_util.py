import os
import shutil

def create_output_folders(directory: str, folder_names: list[str]) -> None:
    """
    Creates folders for the results.

    Parameters:
    - directory (str): The name of the directory in which results folder should be creates.
    - folder_names (list[str]) : Names of the folder for classification.

    Returns:
    - None
    """
    results_directory = os.path.join(directory, "results")
    if os.path.exists(results_directory):
        shutil.rmtree(results_directory)
    os.makedirs(results_directory)
    for folder_name in folder_names:
        folder_name_directory = os.path.join(results_directory, folder_name)
        os.makedirs(folder_name_directory)