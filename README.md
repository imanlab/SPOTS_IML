### Simultanious Prediction of Optical and Tactile Sensation.

This repo accompanies the T-RO 2023 paper: Combining Vision and Touch for Physical Robot Interaction

First Author: Willow Mandil
Second Author: Amir Ghalamzan E

We explore the impact of adding tactile sensation to video prediction models for physical robot interactions. 
Predicting the impact of robotic actions on the environment is a fundamental challenge in robotics. Current methods leverage visual and robot action data to generate video predictions over a given time period, which can then be used to adjust robot actions. However, humans rely on both visual and tactile feedback to develop and maintain a mental model of their physical surroundings. 
In this paper, we investigate the impact of integrating tactile feedback into video prediction models for physical robot interactions. We propose three multi-modal integration approaches and compare the performance of these tactile-enhanced video prediction models. Additionally, we introduce two new datasets of robot pushing that use a magnetic-based tactile sensor for unsupervised learning. The first dataset contains visually identical objects with different physical properties, while the second dataset mimics existing robot-pushing datasets of household object clusters. 
Our results demonstrate that incorporating tactile feedback into video prediction models improves scene prediction accuracy and enhances the agent's perception of physical interactions and understanding of cause-effect relationships during robot interactions. 


## Dataset Description
Locate the 2 datasets and their descriptions at: 
  - https://github.com/imanlab/object_pushing_MarkedFrictionDataset
  - https://github.com/imanlab/

To format the data apply: https://github.com/imanlab/SPOTS_IML/data_formatting/format_data.py

The universal trainer and tester can be used to train and test all the models explored in the paper. Use the input arguments to change for differnet datasets, models, model hyper parameters and more.



