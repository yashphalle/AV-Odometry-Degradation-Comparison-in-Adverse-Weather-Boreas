# AV Odometry Degradation Comparison in Adverse Weather - Boreas

EECE5554 Robotics: Sensing and Navigation - Team 10

LiDAR and visual odometry algorithm degradation under adverse weather (snow, rain) using the [Boreas](https://www.boreas.utias.utoronto.ca/) multi-season autonomous driving dataset. Trajectories are evaluated against accurate GPS/IMU ground truth using ATE and RPE metrics.

## Algorithms

**LiDAR Odometry:** KISS-ICP, LOAM, FLOAM, CT-ICP

**Visual Odometry:** ORB-SLAM3, RTAB-Map

## Dataset Sequences

| Weather | Sequence |
|---------|----------|
| Clear   | `boreas-2020-11-26-13-58` |
| Snow    | `boreas-2021-01-26-10-59` |
| Rain    | `boreas-2021-08-05-13-34` |

## Setup
```bash
python -m ensurepip --upgrade
python -m pip install -r requirements.txt
```

## Download Data
```bash
# By weather label
python data_downloader.py --sequences clear snow rain --output ~/boreas_data

# By sequence name
python data_downloader.py --sequences boreas-2020-11-26-13-58 --output ~/boreas_data

# Specific modalities only
python data_downloader.py --sequences clear --modalities lidar applanix calib

# Preview without downloading
python data_downloader.py --sequences clear snow rain --dry-run
```