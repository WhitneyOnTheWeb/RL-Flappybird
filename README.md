## REQUIREMENTS
Visual Studio Code will be used as the IDE for this project, though most any text editor will work for reproducing it. The project has the following dependencies:

  -  Git
  -  Python 3 environment with the following packages installed:
    -  PyGame; open source game development platform for Python
    -  Scipy; open source scientific computing package suite for Python
    -  Scikit-Learn; popular machine learning library for Python
    -  Keras; expansive open source library for neural networks
    -  TensorFlow; suite of open source deep learning libraries
    -  OpenCV; open source image processing library

Installing a Python distribution and creating the environment will be done through Anaconda. To automatically create a flappy environment preconfigured with the correct dependencies for this project:

  1.	Download the requirements file `[environment.yml](https://raw.githubusercontent.com/WhitneyOnTheWeb/RL-Flappybird/master/environment.yml)` to a `{path}` on your system
  2.	Edit  environment.yml so prefix matches your Anaconda `\envs` path
  3.	From Anaconda Prompt run:
    `conda env create -f {path}\environment.yml`

# EMULATION#
There are several widely available opensource codebases in existence that emulate Flappy Bird and provide all necessary assets. This project will utilize a popular clone FlapPy Bird , built in Python using pygame and available on GitHub under the MIT License , which provides all functionality and should require relatively little code modification to make it suitable for an RL agent.

1.	From a command terminal cd to the project directory and run:
    `git clone https://github.com/sourabhv/FlapPyBird.git`
2.	All graphics are located in the `\assets` folder
3.	The game can be launched by running `flappy.py` from your Python terminal 
4.	The **↑** or **Space** keys are used to flap in lieu of tapping on the screen
