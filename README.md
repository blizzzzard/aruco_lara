# aruco_lara

## Installation

Requirements installation.

```bash
pip install -r requirements.txt
```

This aruco lib has four tasks that can be executed:

## Markers generation

```bash
python main.py -t generation -s 512 -n 1
```

## Markers detection

```bash
python main.py -t detection
```

## Camera calibration

For camera calibration is necessary a paper print of 'calibration pattern.png'. And then images are captured using the camera to be calibrated:

```bash
python main.py -t calibration -c ./calib_data
```

Press 's' to save image and 'q' to quit.


## Pose estimation

```bash
python main.py -t estimation -c ./calib_data
```

## acknowledgment

Tasks fuction were taken from: https://github.com/SiliconJelly/OpenCV/tree/main/Distance%20Estimation
