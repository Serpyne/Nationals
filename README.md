# Nationals

## We are TS.

This project contains the WINNING robot code for the RoboCup National Competition being held in Canberra, on October 10-12th, 2025. \
For any use, please install the library dependencies for Python 3.11.2 through [requirements.txt](requirements.txt) 

# Installation

Firstly, clone the repository onto your Rapsberry Pi 5, then go into the directory.
```
~ $ git clone https://github.com/Serpyne/Nationals.git
~ $ cd Nationals
```
After cloning the repository, initialise a virtual environment with the flag `--system-site-packages`, as the Python environment must have access to the **picamera2 v0.3.30** library.

```
python -m venv --system-site-packages env
```
This process may take a bit of time, so be patient. \
Then, install the dependencies in [requirements.txt](requirements.txt).
```
env/bin/pip install -r requirements.txt
```
Finally, run [robot.py](robot.py) for the main robot script. Ensure that the I2C addresses and ports are all correct.
```
env/bin/python robot.py
```
The file at [ui.py](ui.py) contains the interface for colour, exposure and saturation calibration for the [Raspberry Pi High Quality Camera M12](https://raspberry.piaustralia.com.au/products/raspberry-pi-hq-camera-m12-mount) with the [M12 Fish-Eye Lens](https://core-electronics.com.au/m12-high-resolution-lens-14mp-1846-ultra-wide-angle-272mm-focal-length-compatible-with-raspberry-pi-high-quality-camera-m12.html)

# Robot Script File
We run [robot.py](robot.py) before each match, then a three-way switch selects the different modes: Calibration, Idle, Running
