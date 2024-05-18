# SA-Classification
This is a matlab package for "A phenome-wide association and Mendelian randomization study for suicide attempt within UK Biobank"

"SA-Classification" is a package written in Mtalab and the name stands for a classification model of suicide attempt.

This repository contains the following files:

main.m is the main function of the "SA-Classification"  package.
Model.m is the optimization function of the multi-modal shared latent-space learning model.
UpdateD.m is the function of iteratively updating the projection matrix.

The input are expected to be:

ModalLabel.csv: This file contains the label of the modal to which each feature belongs.
Data: The data should be divided into training sets, verification sets, and test sets, and named accordingly. In each data file, the first column should be the ID of the sample, the second column should be the sample label (0 or 1), followed by the features.

Please contact Meiyan Huang (huangmeiyan16@163.com) or Xiaoling Zhang (zhangxiaoling9911@163.com) for any comments or questions.
