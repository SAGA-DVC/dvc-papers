# A Better use of Audio-Visual Cues: Dense Video Captioning with Bi-modal Transformer

Dense Video Captioning: 
> First localize the events, and then produce one-sentence descriptions of each of them. 

[The paper.](https://arxiv.org/abs/2005.08271)

## Novelty
- ***Bi-modal transformer*** with ***multi-head proposal generator***.
- In the transformer architecture, attention is used to fuse the two input sequences.
- YOLO inspired proposal head for multi-head proposal generator.

#### Dataset: ActivityNet Captions
SOTA performance on BLEU@3-4 and F1 metrics.

## Architecture

![](assets/Pasted%20image%2020210730143821.png)
1. **Input**: continuous features stacked in a sequence.
	1. Visual: Inflated 3D network (I3D)
	2. Audio: VGGish
	3. Tokens: GloVe
2. **Encoder**: Both feature sequences are self-attended and then passed through $N$-layered encoder to produce *bi-modal sequence representations* using novel *bi-modal multi-headed attention* blocks to fuse features from both sequences. 
3. **Proposal Generator**: Uses the representations output from encoder to generate proposals and their confidence scores. 
4. A pre-defined number (top) proposals are selected to **clip** the input feature sequences. 
5. **Encoder**: Re-represents clipped features (since some features have been removed, a representation needs to be calculated again).
6. **Decoder**: The encoder's outputs are passed to the bi-modal attention blocks in every decoder layer, along with the representation of previously generated caption words.
7. **Caption Generator**: Last layer representation of the decoder is used to generate the next word. 
Start token is used as the first caption token (otherwise it would be empty), and the caption is generated word-by-word until end token is sampled.

## Captioning Module
### Goal
Produce a caption for each proposal given to it.

### Flow
1. Bi-modal Encoder takes in audio feature sequences A and video feature sequences V, which correspond (temporally) to to a proposal.
2. Bi-modal Encoder outputs $A^v$ (video attended audio features) and $V^a$ (audio attended video features).
3. Bi-modal Decoder 
	1. uses previous caption words ($c_1$, $c_2$, $c_3$, ..., $c_{t-1}$), self-attends them, 
	2. then carries out bi-modal attention of $c_{t-1}$ with $A^v$ and $V^a$.
	3. Then there's a bridge (?)
	4. Position-wise (to encode order info) fully-connected layers
4. The final representation of the Decoder is used to model the distribution of the next caption word, over the vocabulary ==> FC layer with softmax.

### Other stuff
There are Normalization layers and dropout at some places

## Event Generation Proposal Module
### Goal
Generate a set of proposals, given a video.

### Flow
It consists of two parts:
- Bi-modal encoder
- Bi-modal multi-headed proposal generator

1. Entire sequence is input to bi-modal encoder
2. In the bi-modal encoder, 
	1. Self-attention for audio, video
	2. $A^v$ and $V^a$ are calculated
3. The final encoder representation of the two modalities $A^v$ and $V^a$ are then given to the Proposal Generator:
	1. The two modalities may have different dimensions with respect to the time duration(sequence length) (does this mean they are not necessarily in sync?). Hence they are not fused here, but instead there are two distinct sets of proposal generator heads, one set for each modality. Hence, for each modality individually, the predictions are made at every time step, forming a "common pool of cross-modal predictions".
	2. Inspired by YOLO and RPN ([Convolutional layers with anchor boxes](https://arxiv.org/abs/1612.08242)):
        1. Anchors, calculated apriori, by running K-Means Clustering on the ground truth event lengths (1D). Each centroid of a cluster is taken as an anchor in the anchor set $\psi$.  The distance metric used is the Euclidean distance, while YOLO uses IoU.
	    2. Predictions: Temporal boundaries, and confidence scores are found by the three values predicted by proposal head: center, length, confidence. Temporal boundaries are calculated using center and length. 
	    3. $$center = p + \sigma(c)$$
            $$length = anchor\_length \cdot exp(l)$$
            $$confidence = \sigma(o)$$
            Here, $p$ is the position of grid cell (position in sequence). The network predicts $\sigma(c)$, $l$ and $\sigma(o)$. Refer the [YOLOv2 paper](https://arxiv.org/abs/1612.08242); this is the 1D version of it. The sigmoid function constraint is used for the center so that network is more stable and learning becomes easier.
	1. Select the top 100 (by confidence score) from common pool of proposals.


## Training
1. Captioning module is trained with ground truth proposals
2. Proposal Generator is trained using the now trained bi-modal encoder from captioning modules
3. Loss: KL-divergence, with "Label Smoothing".
4. Each proposal head uses MSE for localization and cross-entropy for objectness (proposal or not) loss. Ref YOLO.