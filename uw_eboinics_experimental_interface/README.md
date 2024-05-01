[![DOI](https://zenodo.org/badge/268858214.svg)](https://zenodo.org/badge/latestdoi/268858214)


# EEG online experiment GUI

## Description

This is a GUI designed for EEG data online collection and experiment. Most commercial EEG 
instruments such as NeuroElectric, Gtec and BrainProducts are compatible with this GUI as 
long as they are supported by [lab streaming layer (LSL)](https://labstreaminglayer.readthedocs.io/info/supported_devices.html) . The basic communication with LSL is done by [1], 
this app expands the GUI to suit EEG/EMG/ECG experiments.

This GUI provides experimenters a ready-togo platform with following functions:
1. Recording
2. Experiment protocol DIY
3. Online feedback for experimenter
    1. EEG signal online visualization
    2. Interested signal extraction and visualization 
    3. Bad trial monitor
4. Online feedback for subject
    1. Task image
    2. Task sound
<br>

![Alt text](docs/tutorial_images/Exp_record.png?raw=true)

## New feature!
Try [Replay recorded file](https://willsniu186.github.io/EEG-Online-Experiment-GUI/build/html/Replay.html) with sample data even if you don't have devices around.

## Getting Started


## [Installation](https://willsniu186.github.io/EEG-Online-Experiment-GUI/build/html/Installation_and_setup.html)

## [Tutorial](https://willsniu186.github.io/EEG-Online-Experiment-GUI/build/html/Tutorial.html)

### [Outline](https://willsniu186.github.io/EEG-Online-Experiment-GUI/build/html/Tutorial_outline.html)

### [For users](https://willsniu186.github.io/EEG-Online-Experiment-GUI/build/html/Tutorial_for_users.html)

   1. [Installation and setup](https://willsniu186.github.io/EEG-Online-Experiment-GUI/build/html/Installation_and_setup.html)

   2. [Replay recorded file](https://willsniu186.github.io/EEG-Online-Experiment-GUI/build/html/Replay.html)

   3. [Installation and setup](https://willsniu186.github.io/EEG-Online-Experiment-GUI/build/html/Installation_and_setup.html)

   4. [Connect with devices](https://willsniu186.github.io/EEG-Online-Experiment-GUI/build/html/Connect_with_device.html)

   5. [Open GUI](https://willsniu186.github.io/EEG-Online-Experiment-GUI/build/html/Open_GUI.html)

   6. [Subject Information Input](https://willsniu186.github.io/EEG-Online-Experiment-GUI/build/html/Sub_info.html)

   7. [Experimental Protocol Design](https://willsniu186.github.io/EEG-Online-Experiment-GUI/build/html/Exp_protocol_design.html)

   8. [Event Number](https://willsniu186.github.io/EEG-Online-Experiment-GUI/build/html/Event_number.html)

   9. [Oscilloscope](https://willsniu186.github.io/EEG-Online-Experiment-GUI/build/html/Scope.html)

   10. [Online Monitor](https://willsniu186.github.io/EEG-Online-Experiment-GUI/build/html/Online_monitor.html)

   11. [Start Experiment](https://willsniu186.github.io/EEG-Online-Experiment-GUI/build/html/Start_exp.html)

   12. [After Experiment](https://willsniu186.github.io/EEG-Online-Experiment-GUI/build/html/After_exp.html)

### [For developers](https://willsniu186.github.io/EEG-Online-Experiment-GUI/build/html/Tutorial_for_developers.html)

   1. [Understand Code Structure](https://willsniu186.github.io/EEG-Online-Experiment-GUI/build/html/Understand_the_code_structure.html)

   2. [Design Your Own GUI](https://willsniu186.github.io/EEG-Online-Experiment-GUI/build/html/Design_your_own_GUI.html)

   3. [Add Your Own Device](https://willsniu186.github.io/EEG-Online-Experiment-GUI/build/html/Add_your_own_device.html)

## Best Practice

The processing speed of GUI depends on your computer specs, there might be some irregular timer issue if your CPU and RAM is out of date. The tested machine has 16GB RAM and Core i7 CPU, which was enough for my experiment. However, some practice twicks might help you reduce the effect:
1. Reduce oscilloscope time span. 
2. Reduce oscilloscope channels to plot.

32 channels with 10 seconds time span on oscilloscope during recording worked fine for me.


## Contact
Jiansheng Niu: jiansheng.niu1@uwaterloo.ca

## Contributors
Jiansheng Niu: jiansheng.niu1@uwaterloo.ca
Aravind Ravi: aravind.ravi@uwaterloo.ca
Hyowon Lee: hyowon.lee@uwaterloo.ca

_Developed in [Ebionics lab](https://uwaterloo.ca/engineering-bionics-lab/), University of Waterloo, Canada._
## Citation
[1] https://github.com/dbdq/neurodecode.git

