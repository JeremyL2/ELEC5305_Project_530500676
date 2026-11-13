# ELEC5305_Project_530500676
Major Project for ELEC5305 Acoustics, Signal and Speech Processing

This project investigates how overlapping underwater vessel sounds affect a pretrained ResNet50 classifier’s ability to recognise cargo, passenger, tanker and tug vessels. Controlled mixtures of DeepShip recordings vary relative source level, overlap duration and temporal placement to assess source detection, detection retention and multilabel performance. Evaluation on the independent PORTIA dataset examines cross-dataset generalisation. The research characterises the classifier’s limitations when recognising simultaneous vessel classes and transferring between recording environments.

## Credits

DAT255 is the pretrained model used in the study.

Original project: https://github.com/voibit/DAT255
Author: Jan-Kristian Mathisen
Copyright (c) 2022 Jan-Kristian Mathisen
License: MIT License

The original MIT License is available at:
https://github.com/voibit/DAT255/blob/main/LICENSE


Deepship is the main dataset used in this study.
DeepShip Dataset:
    Irfan, M., Jiangbin, Z., Ali, S., Iqbal, M., Masood, Z., & Hamid, U. (2021).
    DeepShip: An underwater acoustic benchmark dataset and a separable
    convolution based autoencoder for classification.
    Expert Systems with Applications, 183, 115270.
    DOI: https://doi.org/10.1016/j.eswa.2021.115270

Dataset repository: https://github.com/irfankamboh/DeepShip


PORTIA is the a supplementary dataset also used.

PORTIA Dataset:
    Fernández García, A., Belmonte-Hernández, A., Higón Sorribes, J. M.,
    Company, R., Pérez Carrasco, F., Álvarez García, F., & Gutiérrez, Á.
    (2026). PORTIA: A Large-Scale Port Acoustic Dataset with AIS-Grounded
    Annotations for AI-Based Vessel Monitoring. Scientific Data.
    DOI: https://doi.org/10.1038/s41597-026-08384-9

    Dataset: PORTIA, version 2
    Dataset DOI: https://doi.org/10.5281/zenodo.21409381



## Instructions
Main notebook is in dat255_pilot called DeepShip_Overlap_Experiments.ipynb
The same folder also contains wrappers to run the pretrained model on CPU only.

Recordings from Deepship and the ResNet50 model from DAT255 are required.
