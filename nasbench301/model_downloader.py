"""
Downloads models required for nasbench so they can be used by example.py
Automatically deletes the zip folders after downloading

Note: 'string {}'.format(arg) used to keep backward compatibility
"""
import os
import sys
from zipfile import ZipFile

import requests
from tqdm import tqdm

# Download URL
# Note: Update if the download location changes
URL_MODELS_1_0 = 'https://ndownloader.figshare.com/files/24992018'
URL_MODELS_0_9 = 'https://ndownloader.figshare.com/files/24693026'

def download(url, path):
    # Taken from: https://stackoverflow.com/a/37573701
    response = requests.get(url, stream=True)

    # Measured in Bytes
    file_size = int(response.headers.get('content_length', 0))
    block_size = 1024
    progress_bar = tqdm(total=file_size, unit='iB', unit_scale=True)

    # Write the download to file
    with open(path, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)

    progress_bar.close()
    if file_size not in (0, progress_bar.n):
        print('Error downloading from {}'.format(url))
        sys.exit(1)

def download_models(version, delete_zip=True,
                    download_dir=os.getcwd()):

    # Create paths and names
    download_url = URL_MODELS_0_9 if version == '0.9' else URL_MODELS_1_0
    zip_filename = 'models_{}.zip'.format(version)
    models_folder = 'nb_models_{}'.format(version)

    current_dir = download_dir
    zip_path = os.path.join(current_dir, zip_filename)
    models_dir = os.path.join(current_dir, models_folder)

    # Check if already exists
    if os.path.exists(models_dir):
        print('Models {} already at {}'.format(version, models_dir))
    else:
        print('Downloading models {} from {} to {}'.format(version,
                                                           download_url,
                                                           zip_path))
        download(download_url, zip_path)

        # Zip contains a folder called 'nb_models' so we just unzip
        # it to the current dir and then rename it to give it a version
        print('Extracting {} to {}'.format(zip_filename, models_dir))
        with ZipFile(zip_path, 'r') as zipfile:
            zipfile.extractall(current_dir)
        unzipped_folder_name = os.path.join(current_dir, 'nb_models')
        os.rename(unzipped_folder_name, models_dir)

        # Finally, remove the zip
        # If the library is used by a different library, these zips
        # would end up taking space in the virtual env where the libray user
        # is unlikely to know they even exists there taking up space
        if delete_zip:
            print('Deleting downloaded zip at {}'.format(zip_path))
            os.remove(zip_path)

if __name__ == "__main__":
    # Parse args
    # Note: Would probably be easier to use a lib for this
    #       Also doesn't give a download arg this way
    version = '1.0'  # default to use 1.0
    if len(sys.argv) == 2:
        if version not in ('0.9', '1.0'):
            print('Usage: python {} {}'.format(sys.argv[0], '[0.9 | 1.0]'))
            sys.exit(1)
        else:
            version = '0.9' if sys.argv[1] == '0.9' else '1.0'

    elif len(sys.arv) > 2:
        print('Usage: python {} {}'.format(sys.argv[0], '[0.9 | 1.0]'))
        sys.exit(1)
    download_models(version, delete_zip=True)
