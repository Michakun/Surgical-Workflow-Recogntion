# Surgical Workflow Recogntion
## Challenge description

Surgical Workflow Recognition for the PEg TRAnsfer Workflow recognition by different modalities (PETRAW) subchallenge (MICCAI 2021).

The challenge provides a unique dataset for online automatic recognition of surgical workflow
of a peg transfer training session. The objective of peg transfer session is to transfer 6 blocks from the left to the right and back (Fig.1). Each block must be extracted from a peg with one hand, transferred to the other hand, and inserted in a peg at the other side of the board.

The dataset contains video, kinematic, and segmentation data for 90 training sequences, and 60 test sequences. The workflow annotation, for each timestamp, consists in a 4-label vector, containing the labels for the current phase, step, left-hand and right-hand action verbs.

Participants are challenged to recognize all levels of granularity of the surgical workflow (phases, steps, and action verbs of both hands) with different modalities configurations, and can submit results for uni-modality-based models (video-based model, kinematic-based model or semantic segmentation-based model) and multi-modality-based models (video + kinematic based model or video + kinematic + semantic segmentation-based model). Each model has to be able to recognize all levels of granularity at once.

The algorithms must be applicable for online applications, i.e., future frames can’t be used for workflow prediction.

![alt text](https://github.com/Michakun/Surgical-Workflow-Recogntion/blob/master/Images/PETRAW.PNG)

## Methodology description

Based on the state-of-the-art study that was conducted, and in order to establish a baseline for the PETRAW challenge, a model combining a feature extractor and a temporal network was the first choice when using video data as input.

![alt text](https://github.com/Michakun/Surgical-Workflow-Recogntion/blob/master/Images/Architecture.PNG?raw=true)
