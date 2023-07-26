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
    
3. Download the data and unzip the files in the `data` folder (For access to the full dataset. For the demo skip steps 3 and 4, and jump straight to step 5.)
   ```
   wget -P data/ http://babeta.ufa.cas.cz/dpisa/down/europlanet/chorus_part1.zip &&
   wget -P data/ http://babeta.ufa.cas.cz/dpisa/down/europlanet/chorus_part2.zip
   ```
   
4. Run data preparation: 
    ```
    python dataprep.py
    ```
5. To download the data for demo: `git lfs pull`
   
6. Run the experiments in Jupyter notebooks under `notebooks`

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
│   ├── 01-Annotation.ipyn                            # Annotate the images using SAM box prompt
│   ├── 02-Modelling-baseline-random_sampling.ipynb   # Baseline experiments with data sampled randomly
│   ├── 03-Modelling-SAM-distillation.ipynb           # Fine tuning SAM via training domain-specific decoder
│   ├── 02-Modelling-Active-Learning.ipynb            # Active Learning experiments with various acquisition functions
├── (sam)                                             # To contain the downloaded SAM weights
├── (src)                                             # the main source code directory
│   ├── datasets.py                                   # custom dataset classes and wrappers
│   ├── models.py                                     # custom models for segmentation
│   ├── strategies.py                                 # contains various Active Learning acquisition functions
│   ├── utils.py                                      # misclessaneous helper functions for all experiments
├── codemeta.json                                     # code metadata
├── conda_packages.txt                                # required conda packages
├── dataprep.py                                       # Python script to convert spectrograms to images and save them for use throughout
├── LICENSE.txt                                       # the full license which this project employs
├── metadata.yml                                      # project metadata
├── pip_requirements.txt                              # required pip packages
├── README.md                                         # this README file

```
### Acknowledgement

<img src="logo.jpg" align="left" width="200px"/>Europlanet 2024 RI has received funding from the European Union’s Horizon 2020 research and innovation programme under grant agreement No 871149.

<br clear="left"/>
