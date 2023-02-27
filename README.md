### Simultanious Prediction of Optical and Tactile Sensation.

This repo accompanies the NeurIPS 2022 paper: Improved Video Prediction with Visual and Tactile Sensation

First Author: Willow Mandil
Second Author: Amir Ghalamzan E

Enabling a robotic agent to predict the effect of its actions on its environment is a core robotics challenge. Existing methods apply visual and robot state/action information to make video predictions. However, the less observable the state of physical robot interaction is through visual information, the more uncertain the model is in predicting those interactions. We believe the introduction of tactile sensation is essential. Although it is possible to estimate the forces and dynamics involved in interactions through computer vision, it may be unreliable when presented with real-world interactions, e.g. in the case of occluded camera view. Tactile sensation has not yet been integrated into visual prediction systems for physical robot interaction. This paper explores the impact of introducing this second sensing modality to video prediction of physical robot interactions. We first present a robot pushing dataset of 45'000 frames of a visually identical block object with different friction properties. We then introduce three key video prediction architectures. We explore these models and their key features in a comparative study to show that tactile sensation improves scene prediction accuracy and cause-effect understanding during physical robot interactions.

## Dataset Description
Locate the dataset and its description at: https://github.com/imanlab/object_pushing_MarkedFrictionDataset

To format the data apply: https://github.com/imanlab/object_pushing_SPOTS_NIPS/blob/main/data_formatting/format_data.py 


## Model Architectures:
For the prediction models p and q we use the convolutional dynamic neural advection (CDNA) model described in finn2016unsupervised with the additional stochastic variational inference network described in babaeizadeh2017stochastic. Of course, the focus of this work is to show the effect on video prediction accuracy of introducing tactile data, so it in not essential to integrate with the best performing VP models at the time, however the model described appears to be one of the best in this class at time of writing nunes2020action. 

These two models are therefor benefiting from one another, this holistic approach enables the scene prediction to be enhanced by tactile sensation, and the tactile prediction to be enhanced by the visual scene.

<img src="https://github.com/imanlab/object_pushing_SPOTS_NIPS/blob/main/SVTG.png" height="450">

<img src="https://github.com/imanlab/object_pushing_SPOTS_NIPS/blob/main/SPOTS.png" height="450">

<img src="https://github.com/imanlab/object_pushing_SPOTS_NIPS/blob/main/SPOTS_SVG_ACTP.png" height="450">



