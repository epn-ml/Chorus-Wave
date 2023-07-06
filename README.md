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
   pip install numpy matplotlib torch torchvision segment-anything-model jupyter-bbox-widget
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

TDB
