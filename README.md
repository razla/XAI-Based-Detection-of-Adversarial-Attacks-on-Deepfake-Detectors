# Attack Detection On Deepfake Detector

[Paper](https://arxiv.org/abs/2403.02955)

## Setup Requirements

The code is based on PyTorch 2.0.1 with CUDA 11.7 and requires Python 3.9
The code requires the dlib library. The wheel of this requirement is provided wheels directory only for Windows operating system. For other operating systems, you will need to compile the dlib library from source.


Install requirements via ```pip install -r requirements.txt```

## Dataset
To download the FaceForensics++ dataset, you need to fill out their google form and and once accepted, they will send you the link to the download script.

Once, you obtain the download link, please head to the [download section of FaceForensics++](https://github.com/ondyari/FaceForensics/tree/master/dataset). You can also find details about the generation of the dataset there. To reproduce the experiment results in this paper, you only need to download the c23 videos of Deepfakes generation method.


## Victim Pre-trained Deepfake Detectors Models

### XceptionNet
You can find the pre-trained XceptionNet model weights in 'models' directory in file name 'xception.p'.

### EfficientNetB4ST

You can find the pre-trained EfficientNetB4ST model weights in 'models' directory in file name 'EfficientNetB4ST.pth'.
    


## Running an attack on videos directory

This setup is for running pgd, fgsm, nes attack to create adversarial examples on video files in directory. 
```shell
python attack.py
-i <path to input folder of videos with extenstion '.mp4' or '.avi'>
-mi <path to pre-trained model weights>
-mt <type of model, choose either xception or EfficientNetB4ST >
-o <path to output folder, will contain output videos >
-a <type of attack, choose from the following: pgd, fgsm, nes >
--eps <epsilon value for the attack >
--cuda < if provided will run the attack on GPU >
```
Example:
```shell
python attack.py -i Data/DFWebsite/DeepFakes/c23/videos/ -mi models/xception.p -mt xception -o temadv/ -a pgd --cuda --eps 0.01
```

This setup is for running apgd, square attack to create adversarial examples on video files in directory.
```shell
python auto_attack.py
-i <path to input folder of videos with extenstion '.mp4' or '.avi'>
-mi <path to pre-trained model weights>
-mt <type of model, choose either xception or EfficientNetB4ST >
-o <path to output folder, will contain output videos >
-a <type of attack, choose from the following: apgd-ce, square >
--eps <epsilon value for the attack >
--cuda < if provided will run the attack on GPU >
```
Example:
```shell
python auto_attack.py -i Data/DFWebsite/DeepFakes/c23/videos/ -o temadv/ -mt EfficientNetB4ST -mi models/EfficientNetB4ST.pth -a apgd-ce --cuda --eps 0.01
```

## Detect frames of videos in a directory
This setup is for detecting frames of the videos in a directory. The output of this setup is json file with all probs of the frames of the videos. This json will be use for calculate accuracy of the deepfake detector.
```shell
python detect_from_video.py
--video_path <path to directory containing videos>
--model_path <path to deepfake detector model>
--model_type <type of model, choose either xception or EfficientNetB4ST >
--output_path <path to output directory, will contain output frames >
--cuda < if provided will run on GPU >
```
Example:
```shell
python detect_from_video.py --video_path tempadv/attacked/apgd-ce/EfficientNetB4ST --model_path models/EfficientNetB4ST.pth --model_type EfficientNetB4ST --output_path temadv/attacked/apgd-ce/EfficientNetB4ST --cuda
```

## Calculate statistics of the videos in a directory
After creating adversarial examples or detecting frames of the videos with ```detect_from_video.py``` script, you can calculate the statistics of the videos in a directory. With this script we can calculate accuracy of the deepfake detector.
```shell
python summarize_stats.py
--dataset_path <path to directory containing videos and their json files >
--model_type <type of model, choose either xception or EfficientNetB4ST >
--threshold <threshold value for consider fake or real >
```
Example:
```shell
python summarize_stats.py --dataset_path tempadv/attacked/apgd-ce/EfficientNetB4ST/ --model_type EfficientNetB4ST --threshold 0.5
```

## Create XAI maps for test or train set

this setup is for creating XAI maps for the test or train set of videos in a directory.

```shell
python create_xai.py
--video_path <path to directory containing videos>
--model_path <path to deepfake detector model>
--model_type <type of model, choose either xception or EfficientNetB4ST >
--output_path <path to output directory, will contain output frames >
--cuda < if provided will run on GPU >
--xai_methods <list of xai methods to use, choose from the following: GuidedBackprop, Saliency, InputXGradient, IntegratedGradients >
```
Example:
```shell
python create_xai.py --video_path tempadv/attacked/apgd-ce/EfficientNetB4ST --model_path models\EfficientNetB4ST.pth --model_type EfficientNetB4ST --output_path Frames/attacked/<attack_name> --cuda --xai_methods GuidedBackprop Saliency InputXGradient IntegratedGradients
```
The output will be is a directory containing facecrops and directory for each XAI method containing the XAI maps for each frame of the video.

Example of the directory structure of videos detected with EfficientNetB4ST deepfake detector:
```
<output path>/
    - EfficientNetB4ST/
      - Frames/
      - GuidedBackprop/
      - Saliency/
      - InputXGradient/
      - IntegratedGradients/
```

## Train an Attack Detector
For training an attack detector, you can use the following setup:
```shell
python train.py
-tri <path to input folder of train real frames (facecrop + XAI maps) >
-tai <path to input folder of train attacked frames (facecrop + XAI maps) >
-vri <path to input folder of validation real frames (facecrop + XAI maps) >
-vai <path to input folder of validation attacked frames (facecrop + XAI maps) >
-e <number of epochs>
-lr <learning rate>
-b <batch size>
-fr <boolean, add if freeze backbone layers on training>
-xm <XAI method to use, choose from the following: GuidedBackprop, Saliency, InputXGradient, IntegratedGradients>
-dt <type of deepfake detector, choose either xception or EfficientNetB4ST >
```
The output of the training summery and the models weights will be saved in ```runs_resnet50_frozen<-fr>/<Deepfake Detector type>/<XAI method>/<Data and time of the training>/``` directory.
There are 2 kind of models that will be saved, one is the model with the best validation accuracy (best_model.pth) and the other is the model at the last epoch (last_model.pth).

Example:
```shell
python train.py -tri Frames/real/ -tai Frames/attacked/apgd-ce/EfficientNetB4ST/ -vri Frames/real/ -vai Frames/attacked/apgd-ce/EfficientNetB4ST/ -e 100 -lr 0.001 -b 16 -xm GuidedBackprop -dt EfficientNetB4ST
```

## Test an Attack Detector
For testing an attack detector, you can use the following setup:
```shell
python test.py
-mi <path to pre-trained attack detector model weights>
-mt <type of deepfake detector model, choose either xception or EfficientNetB4ST >
-xm <XAI method to use, choose from the following: GuidedBackprop, Saliency, InputXGradient, IntegratedGradients>
-o <path to output folder>
-rd <path to directory containing real frames (facecrop + XAI maps) >
-ad <path to directory containing attacked frames (facecrop + XAI maps) >
-b <batch size>
```
The output of the testing will be saved in ```<output path>\<deepfake detector type>\<XAI method>``` directory. The directory will contain the average accuracy on the test set in file ```model_acc.txt```, and the ROC curve graph in file ```ROC_graph.png```.

Example:
```shell
python test.py -mi runs_resnet50_frozenFalse/EfficientNetB4ST/GuidedBackprop/2022-03-29_14-00-00/best_model.pth -mt EfficientNetB4ST -xm GuidedBackprop -o test_results/ -rd Frames/real/ -ad Frames/attacked/apgd-ce/EfficientNetB4ST/ -b 16
```