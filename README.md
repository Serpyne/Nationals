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

## Reflection
Team TS has gone through a two-year journey, starting in Feburary 2024 up until October 2025.

In 2024, my team used a Raspberry Pi 5 with a PiCamera Module 3 for computer vision, and computing the ball angle and distance to send to a Raspberry Pi Pico (via UART serial data) which handled the ball chasing logic and controlling of motor signals. The setup was limited by an unstable communication between the two microcontrollers, whose data was corrupted by either the power that the motors required when enabled or interference from the other I2C components; we never figured it out.

Regardless, we were able to win Victorian States Open Soccer 2024 in an intense grandfinal against JMSS' only team, albeit being rather hilarious at times since more than half of the game was spent watching both the home and away teams miss the ball or drive in circles to find it.

Thus, at the beginning of 2025, we decided to revamp our entire setup by removing the Raspberry Pi Pico and voiding the redundancy of the UART connection, whose corrupted data make gameplay **completely** impossible. We also moved from a Python -> C/C++ environment to completely Python so that the vision and motor modules operated at the same refresh rate. We in turn found success with this setup as there were no issues meshing all of the modules together, and the blunt of problems arose from hardware issues (dribbler sucked, baseplate was too low so it got stuck on certain parts of the field, robot walls to high so when the ball was flush against the robot, our camera could not see any orange).

For the majority of development we had a working movement system (moving the baseplate up provided better contact with the ground and so the robot wouldn't lose orientation) and a sufficiently adequate dribbler (ball would slip out when it rotated). Then, in the rush to get a robot working for the nationals competition in October, our team members worked day and night to put the solenoid and bluetooth modules in and get everything working. There would be errors that were un-reproducable that made development **extremely** annoying, but we eventually got it done and came out undefeated after two days of competition in Canberra.
