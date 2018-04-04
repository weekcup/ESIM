"""
Download the data necessary for the ESIM model:
    - Stanford Natural Language Inference (SNLI) dataset.
    - GloVe word embedding vectors.
"""
# Aurelien Coet, 2018.

import os
import sys
import zipfile


# Function from https://github,com/lukecq1231/nli/blob/master/data/download.py
def download(url, targetdir):
    """
    Download data from an url and save it in some target directory.
    (Note: wget must be installed on the machine in order for this function to
    work.)

    Args:
        url: The url from which the data must be downloaded.
        target_dir: The target directory where the downloaded data must be
                    saved.

    Returns:
        The path to the downloaded data file.
    """
    filename = url.split('/')[-1]
    filepath = os.path.join(targetdir, filename)
    print("* Downloading data from {}".format(url))
    os.system("wget {} -O {}".format(url, filepath))
    return filepath


# Function from https://github,com/lukecq1231/nli/blob/master/data/download.py
def unzip(filepath):
    """
    Unzip a zipped file.

    Args:
        filepath: The path to the file to unzip.
    """
    print("* Extracting: {}".format(filepath))
    dirpath = os.path.dirname(filepath)
    with zipfile.ZipFile(filepath) as zf:
        zf.extractall(dirpath)
    os.remove(filepath)


def download_unzip(url, targetdir):
    """
    Download and unzip data from an url and save it in a target directory.

    Args:
        url: The url to download the data from.
        targetdir: The target directory in which to download and unzip the
                   data.
    """
    filepath = os.path.join(targetdir, url.split('/')[-1])
    if not os.path.exists(targetdir):
        os.makedirs(targetdir)
    # Download and unzip if the target directory is empty.
    if not os.listdir(targetdir):
        unzip(download(url, targetdir))
    # Skip downloading if the zipped data is already available.
    elif os.path.exists(filepath):
        print("* Found zipped data - skipping download")
        unzip(filepath)
    # Skip unzipping if the unzipped data is already available.
    else:
        print("* Found unzipped data - skipping download and unzipping")


if __name__ == "__main__":
    datadir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                           "..", "data")
    snli_url = "https://nlp.stanford.edu/projects/snli/snli_1.0.zip"
    glove_url = "http://www-nlp.stanford.edu/data/glove.840B.300d.zip"
    print(20*'=' + "Fetching the SNLI data:" + 20*'=')
    download_unzip(snli_url, os.path.join(datadir, "snli"))
    print(20*'=' + "Fetching the GloVe data:" + 20*'=')
    download_unzip(glove_url, os.path.join(datadir, "glove"))
