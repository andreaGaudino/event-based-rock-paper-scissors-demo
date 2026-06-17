# Event-Based Roshambo Demo

This repository contains a desktop-based interactive demo of the "Rock-Paper-Scissors" game using an event camera.


<img width="800" alt="dextra-poster-1" src="https://github.com/user-attachments/assets/40027abc-2474-4e91-8186-a11c2b9479df" />

## Project Overview
This project was developed as part of the **ResProj (Research Project) course at Eurecom**. 

The goal of this work is to port and adapt the original gesture recognition pipeline developed for the [Dextra Roshambo](https://github.com/SensorsINI/dextra-roshambo-python) project into a real-time desktop application. By leveraging the low latency of event-based vision, the system can anticipate human gestures and respond instantly.

### Key Modifications:
* Migrated the event processing pipeline to the modern **DV software** framework.
* Developed a custom GUI using **Python OpenCV** for real-time visualization and user interaction.
* Optimized the inference loop for desktop performance, removing external hardware dependencies (e.g., robotic arms).
* Fine-tuned the original model, as it was unable to detect gestures when the hand is in vertical orientation.

## Project Structure

```text
.
├── assets/
|    ├── sample_frames/
|    │   └── *.png                   # Image frames (0000.png to 0359.png)
|    ├── symbols/
|    │   ├── beat dextra poster.pdf
|    │   ├── dextra-icon.png
|    │   ├── dextra-icon.xcf
|    │   ├── dextra-poster.pdf
|    │   ├── paper.* # .ai, .png, .psd files
|    │   ├── rock.* # .ai, .png, .psd files
|    │   └── scissors.* # .ai, .png, .psd files
├── model/
│   ├── numpy_weights/
│   │   └── *.npy               # Layer weights, biases, and shifts
│   ├── variables/
│   │   ├── variables.data-00000-of-00001
│   │   └── variables.index
│   ├── dextra_roshambo.tflite
│   ├── dextra_roshambo_finetuned.h5
│   ├── finetuned_model_dextra_roshambo.tflite
│   └── roshambo.h5
├── src/
|    ├── fine_tuning/
|    │   ├── aedat_npy_converter.py
|    │   └── fine_tuning_model.py
|    ├── main.py
|    ├── read_camera.py
|    ├── test_images.py
|    └── utils.py
├── .gitignore
├── README.md
└── requirements.txt

```


## Setup


Create a Miniconda/Anaconda environment as it is possible to install Python versions that are even not present on your PC, as an alternative you can use venv but you must use **Python 3.9**. If you do not have yet installed both of them: https://www.anaconda.com/download (Miniconda is suggested as it is way lighter).

Once installed: 
###
    conda create -n demo_roshambo python=3.9 
    conda activate demo_roshambo

#### More recent Python versions do not support some of the libraries we will need.

To install the libraries: 
###
    pip install -r requirements.txt



## Event-Camera Setup

To connect and stream data from the event-camera, you will need to install and configure the iniVation DV software.

### 1. Install DV Software
Download and install the DV software by following the instructions in the official documentation:
[DV Software Installation Guide](https://docs.inivation.com/software/dv/gui/install.html)

### 2. Configure the TCP Connection
In order to transfer event data from the DV software to this demo, you need to set up a network connection:
* Open the DV software.
* Create a **TCP connection node**.
* Assign an **IP address** and a **Port number** to this node.

### 3. Update Project Settings
Once your TCP node is configured, you need to link it to the project's code:
* Open the `utils.py` file.
* Update the file with the exact **IP address** and **Port number** you set in the DV software.

## How to run the code

To run the program, enter the folder where the main.py file resides by writing in the terminal:
###
    cd src

Then:
### 
    python main.py

If you have a recorded video (option available on the DV software) it is possible to keep track of the statistics by running the following command:
### 
    python main.py --recording

And it will save frames and all the movements in the `/statistics` folder.

If you wish to quit the execution, press the 'q' key.

## Acknowledgments
* Original event-based gesture pipeline developed by the [SensorsINI team](https://github.com/SensorsINI/dextra-roshambo-python).
* Developed as part of the ResProj (Research Project) course at Eurecom.

## Author
**Andrea Gaudino**
* [LinkedIn Profile](https://www.linkedin.com/in/andrea-gaudino-848487345)
* [GitHub Profile](https://github.com/andreaGaudino)
