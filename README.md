1. To setup the environment:

    ``` 
    conda create --name chorus_wave --file conda_packages.txt
    conda activate chorus_wave
    ```
   Within conda env install the following pip packages:
    ```
   pip install numpy matplotlib torch torchvision segment-anything-model jupyter-bbox-widget
    
   ```
   Install segment-anything-model
    `pip install segment-anything-model`



2. Download weights for SAM model

`mkdir & cd sam` 

`wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth`
