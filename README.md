# Multimedia-Project

## Installation:
* Clone repo (with git client or visual studio)
* Copy [OpenCV(2413) folder](https://www.kuenzler.io/share/opencv.zip) to project root. (same level as src folder)
* Copy the train and test folder from the [INRIA Person Dataset](http://pascal.inrialpes.fr/data/human/) to the project root. If you face problems (e.g. with symlinks on windows) use [these folders](https://www.kuenzler.io/share/Dataset.zip)

* Open Project with visual studio

Project folder structure should look like this:

![project folder](https://www.kuenzler.io/share/mmp_5.PNG "project folder")

The data list files need to be in the root project folder:

![project folder](https://www.kuenzler.io/share/mmp_4.PNG "project folder")

If you face problems, check your project settings:

![project folder](https://www.kuenzler.io/share/mmp1.PNG "project folder")

## Usage:
* Start the programm without any commandline parameters
* You're then able to choose between diffrent modes:
  * (1) Train: Train a new svm. Chose a name you want to use. ".0.xml" and ".1.xml" will be appended to that name for the original and retrained svm. You're then asked to select the number of iterations and a threshold for retraining. You may use the default values. The "disvaluethreshold" is used to gather hard-negatives. Every detection higher (absolute) will be added to the hard-negative feature list. You're then asked if you want to adjust the threshold dynamically. If selected, this will adjust your previsously selected threshold until you get an amout of ~10% hard negatives in relation to the amount of your original negative training data. If you select no, you'll be prompted for approval as soon as the hard negatives are gathered. You may adjust your threshold then and restart the hard-negative mining. Finally you'll be asked if you want to use opencvs auto_train or train function. For most cases, the default option (using auto_train) is the one you want, as the train function only uses the default parameters.
  * (2) Test: Use the complete test set to test your svm. enter the path to your svm you want to use. If you want to compare original and retrained svms, use (5) compare
  * (3) Evaluation: Evaluate a trained svm (original and retrained). Just enter the name as you did in (1), so if you have 1.0.xml and 1.1.xml, you enter just "1". You're then promted for betterdetectionmode. If you select this, all bounding boxes, that have true bounding boxes in them, will be counted as successfully detected. If you select no, it will be evaluated as described in the project manual (with 50% overlap only). The evaluation task will automatically call a pythonscript to create a plot from the gathered data which will not be saved automatically, check "Evaluation script:" for more information.
  * (4) presentation: this mode will chose 10 random pictures from the test set. you'll be promted for an svm name, enter the full name to an original or retrained svm.
  * (5) compare: compare an original svm with it's retrained counterpart. Enter the svm name as in (3), so for comparing 1.0.xml and 1.1.xml, just enter "1". You'll then be able to choose between the full test-set and 10 random images (as in (4)). After selecting, the first image will appear, press enter to see the corresponding second image as well. Press Enter twice to continue to the next picture
  * (6) exit: exit the programm
* Evaluation script: The eval task (3) uses the MMPScript.py to plot the gathered data. therefore you have to have python 2 as well as the dependencys matplotlib and numpy installed (numpy is a dependency of matplotlib, so calling "pip install matplotlib" should do the job. If you want to use the script without the eval task, you need the "detections-<name>.txt" file. you can start the plotgeneration by calling "python MMPScript.py detections-<name>.txt" The plot will not be saved automatically, so remember to save it.
The currently created files all have the svm number "x.0", but refer to both the original and the retrained svm. The true / false in the name indicates wheter the evaluation was called with betterdetection (true) or without (false) (see (3))
  
## Misc
* Try to use self explanatory variable and method names
* Use english names & comments
* if you're using VS 2017, you need the [2015 Build tools](http://landinghub.visualstudio.com/visual-cpp-build-tools). 

