# Chorus wave segmentation in magnetic spectra
This is the repository containing the code for the paper titled ....
```
Citation TBD
```

1. To setup the environment:

   ```
    conda create --name chorus_wave --file conda_packages.txt &&
    conda activate chorus_wave
   ```
   Within conda env install the following pip packages:
   ```
   pip install pip_requirements.txt
   ```
  
2. Download weights for SAM model
    ```
    wget -P sam https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
    ```
    
3. Download the data and unzip the files in the `data` folder
   ```
   wget -P data/ http://babeta.ufa.cas.cz/dpisa/down/europlanet/chorus_part1.zip &&
   wget -P data/ http://babeta.ufa.cas.cz/dpisa/down/europlanet/chorus_part2.zip
   ```
   
5. Run data preparation: 
    ```
    python dataprep.py
    ```
6. Experiment notebooks are contained in `notebooks`

### Project Structure
```
├── (data)                                            # Data directory
│   ├── (npy_1)                                       # To include the unzipped data partition 1
│   ├── (npy_2)                                       # To include the unzipped data partition 2
│   ├── (processed)                                   # Directory to contain processed data images and corresponding annotations
│   │   └── (images)                                  # Includes images created from spectrograms using dataprep.py
│   │   └── (masks+)                                  # Includes the positive annotations created using 01-Annotation.ipynb
│   │   │   └── (train)                               # Train partition
│   │   │   └── (test)                                # Test partition
├── (notebooks)                                       # jupyter notebooks for various experiments
│   ├── 01-Annotation.ipyn                            # notebooks for verifying assumptions about the data
│   ├── 02-Modelling-baseline-random_sampling.ipynb   # notebooks helpful for analyzing models
│   ├── 03-Modelling-SAM-distillation.ipynb           # notebooks helpful for analyzing models
│   ├── 02-Modelling-Active-Learning.ipynb            # notebooks helpful for analyzing models
├── (sam)                                             # To contain the downloaded SAM weights
├── (src)                                             # the main source code directory
│   ├── datasets.py                                   # custom dataset classes and wrappers
│   ├── models.py                                     # custom models for segmentation
│   ├── strategies.py                                 # contains various Active Learning acquisition functions
│   ├── utils.py                                      # misclessaneous helper functions for all experiments
├── codemeta.json                                     # code metadata
├── conda_packages.txt                                # required conda packages
├── dataprep.py                                       # Python script to convert spectrograms to images and save them for use throughout
├── metadata.yml                                      # project metadata
├── pip_requirements.txt                              # required pip packages
├── README.md                                         # this README file :)
├── LICENSE.txt                                       # the full license which this project employs

```
