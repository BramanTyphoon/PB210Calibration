# PB210Calibration
Utility class to perform dating calibration of profiles of Gamma-counted Lead-210 activity

## Motivation
Calculating ages from sediment Pb-210 profiles requires some complicated calculations that are onerous to perform in Excel, the usual platform for those calculations. This module performs those calculations for you, and performs Monte Carlo simulations to develop confidence intervals along the entire profile.

N.B. You should understand the science behind calibration before using this tool, otherwise you may inadvertantly make unfounded assumptions! The python files mention the literature relevant for these calculations.

## Installation
Download the __PB210Calibration.py__ module into a directory on your machine's Python PATH. Then, import into your desired script
```
import PB210Calibration.py
```
That's it!  You can look at and run __PB210Calibration_example.py__ for a canonical example of Lead-210 calibration and to test that the module is working on your machine.

## Files
* [PB210Calibration.py](PB210Calibration.py) - Importable module to perform the calibration
* [PB210Calibration_example.py](PB210Calibration_example.py) - Script containing an example of how to use this class to calibrate a canonical example from the literature.

## Authors
* James Bramante - initial work - [BramanTyphoon](https://github.com/BramanTyphoon)

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

# TODO

* Add tie point capabilities, e.g. from independently calibrated Cs-137 dates



