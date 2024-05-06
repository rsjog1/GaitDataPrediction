## Data Description

This project is based on this [publication](https://research.ece.ncsu.edu/aros/paper-tase2020-lowerlimb/). In our case, we will focus on classifying different locomotion modes based on inertial measurements from individuals. The subjects walked throughout Centennial campus at NC State wearing a device that captured acceloremeter and gyroscope measurements on one of their legs.  The data consists of several trials coming from multiple subjects.

The file containing the inertial measurements is labeled as "Trial001_x.csv" with corresponding labels found in "Trial001_y.csv", where the number corresponds to a trial ID.

Here is a brief description of the files:
  - The "_x" files contain timestamps (first column), and the xyz accelerometers (next three columns) and xyz gyroscope (final three columns) measurements from the lower limb. The time is in seconds and the sampling rate is 40 Hz.
  - The "_y" files contain timestamps (first column) and the labels (second column). (0) indicates standing or walking in hard terrain, (1) indicates going down the stairs, (2) indicates going up the stairs, and (3) indicates walking on soft terrain. The time is is in seconds and the sampling rates is 10 Hz.
