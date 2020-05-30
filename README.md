# EDM_Segmentation_Project
This is a project done by Nathan Lehrer and Zhiguang Eric Zhang for NYU's Music Information Retrieval Class. The project attempts to improve automatic structural segmentation for tracks in the genre of Electronic Dance Music by modifying the beat tracking stage of Oriol Nieto's Music Structure Analysis Framework (MSAF).

This repository contains three pieces of wholly original code:

-- core.py contains the core methods for our project including beat tracking, computation of measure boundaries, and more

-- parameter_tuning.py contains methods we used to optimize our parameters, such as computing the overall tempo error over the dataset

-- main_file.py contains the code used to generate various plots and optimizations invoking the methods in core.py and parameter_tuning.py

This repo also contains a modified version of Oriol Nieto's "featextract.py" which computes beat-synchronous features to be used by segmentation algorithms in MSAF. We modified it to use our beat tracking, first downbeat detection, and other methods to try out different ideas in our project. To use this file, replace "featextract.py" with this one in MSAF.
