# TSP: Temporally Sensitive Pretraining of Video Encoders for Localization Tasks
## The Point
Most localization methods use video features extracted by models that are trained for Trimmed Action Classification (TAC). They're not necessarily suited for Temporal Action Localization (TAL). E.g., R(2+1)D, I3D and C3D have become the *de facto* video feature extracters for TAL, Action Proposal and DVC tasks; these are trained on TAC.

A supervised pre-training paradigm is introduced that also considers background clips (which are not as important as for TAC) and global video information, to *improve temporal sensitivity*. TAC-pretrained features tend to be temporally insensitive.

#### Contributions & Findings:
* TSP trains an encoder to explicitly discriminate between foreground and background clips in untrimmed videos
* Temporally-sensitive features from TSP improves performance for TAL, Action proposal and DVC Tasks
* Consistent performance gains for multiple algorithms/architectures/datasets
* TAL performance is boosted on short action instances.  

### Reasons for most methods using TAC trained video encoders
* Established models
* Impractical to fit untrimmed videos in commodity GPUs without drastically downsampling space or time*

### Pre-training strategy
**Goal: incorporate temporal sensitivity**

Train encoders on the task of
* Classifying foreground clips
* Classifying whether a clip is inside or outside the action.

#### Data
* Untrimmed videos with temporal annotations
* Encoder is trained end-to-end from raw video input
* From an untrimmed video, X is sampled. X: 3 x L x W x H (L = number of frames)
* There is a natural imbalance in the annotations of foreground and background clips, hence, X is sampled in a way so that an equal number from each class is chosen
* Each X is annotated with two labels: 
	* $y^c$: action class label, *if it is from a foreground clip*
	* $y^r$: binary temporal region label, i.e. whether whether the clip is from a foreground/action region ($y^r$ = 1) or background/no-action region ($y^r$ = 0) of the video.* 

####  Local and Global Feature Encoding
* The encoder E transforms a clip X to a local feature f (dim = F).
* Global Video Feature (GVF) is the max-pooled feature from all local features. Different pooling functions considered in Supplementary Material.
* For classifying the temporal region (foreground or background), we combine each local feature f with GVF. So GVF is like the conditional vector (*"given GVF"*).

#### Classification:
Two heads:
1. *F x C* FC layers: for action class label. f --> $y^c$ logits vector (multiclass)
2. *2F x 2* FC layers: for temporal region label. \[f, GVF\] --> $y^r$ logits vector (binary)

![](Pasted%20image%2020210803233321.png)

#### Loss:
1. Cross-entropy for both classification heads
2. If foreground clip, then loss from both heads taken into account, relatively weighted by $\alpha^r$ and $\alpha^c$
3. If background clip, only the temporal region classification loss is taken, weighted by $\alpha^r$

#### Datasets:
* ActivityNet v1.3
* THUMOS14

#### Video encoders:
* ResNet3D
* R(1+2)D

### Results
#### Algorithms used
The algorithms were directly taken from respective implementations, only the input features used were TSP features obtained.
* GTAD: sub-graph localization for Temporal Action Detection
* BMN: Boundary-matching network for Temporal Action Proposal Generation
* BMT: Bimodal Transformer for DVC (Iashin) (audio features kept same)
* P-GCN: Graph CNN for Temporal Action Localization	