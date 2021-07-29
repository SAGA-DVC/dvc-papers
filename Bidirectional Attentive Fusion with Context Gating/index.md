# Bidirectional Attentive Fusion with Context Gating for Dense Video Captioning
Dense Video Captioning: 
> First localize the events, and then produce one-sentence descriptions of each of them. 

#### Application areas:
* Video summarization
* Video retrieval (search)
* Video object detection
* Video segment localization with language query

## Novelty
- ***Bidirectional proposal generator*** to utilize both past and future context while making event predictions
- ***Context Gating*** to balance the contribution from current event and surrounding contexts dynamically
- Solve the *problem of inability to distinguish between events ending at around the same time* by ***attentive fusion of hidden states of proposal generator and the video contents***. It is empirically shown that this attention-fused event representation is at solving this problem compared to the individual (separately taken) methods.
- ***Joint ranking*** for dense captioning during inference stage. Combines proposal score and caption confidence to get high confidence (proposal, caption) pairs, which is the main goal of dense video captioning.

### Single Stream Temporal Action Proposals (SST)
- Better alternative to a simple/naive sliding window method. 
Sliding window method:
	- Classify each window as *action* or *background (non-action / non-event)*
	- Limitation: events limited to predefined sliding window length
	- To overcome this, SST was introduced
- Like sliding window, SST runs through the video only once
- Makes proposal predictions ending at every time step, with *k* different offsets (event length offsets?)
- Problem with SST is again that it does not use future information, which could be valuable towards the task


## Proposal Module (Bidirectional SST)
### Goal
Generate a set of temporal regions (intervals) that possibly contain events. Efficiently encode past, current and future information for proposal generation.

### Flow
1. Input: 
	- Video sequence $X = \{x_1, x_2, x_3, ..., x_L\}$, with L frames
	- Each video frame is encoded using 3D CNN, pre-trained on Sports-1M dataset. Extracted C3D features are *temporal resolution* 16 frames, i.e. each feature ==> 16 frames.
	- PCA is then used to reduce feature dimensionality. Final visual stream: $V = \{v_1, v_2, v_3, ..., v_{L/16}\}$
2. Forward Pass: Input is $V$, in forward order.
	1. LSTM sequentially encodes the visual stream $V$, accumulating visual cues across time. The LSTM hidden states encode visual information os passed time steps.
	2. LSTM hidden states are fed into *k* independent binary classifiers (sigmoid activation, 0-111), giving *k* confidence scores. Each of these scores is of the proposal with end time as current time, and start time as $t - l_i$, where $\{l_1, l_2, l_3, ..., l_k\}$ are the lengths of the predefined *k* proposal anchors.
3. Backward Pass: Input is $V$ in *reverse* order.1
	1. Flow is same as the forward pass, with the classifiers predicting their scores for *k* proposals ***starting*** at current time. So as in the forward pass, we get K proposals and their confidence at every time step.
4. Fusion: Fuse the two sets of proposals by performing a multiplication (Hadamard product, simple multiplication of corresponding proposal scores) of the confidence scores.
5. Proposals with scores larger than a threshold $\tau$ will be selected for captioning.

## Captioning Module
### Goal
Get captions of selected proposals

### Flow
1. Representation of proposals: If only Proposal Module's LSTM's hidden state is taken, the *"discrimination property of event representation is overlooked"*, i.e. the problem of multiple events ending at the same time not being identified as distinct. There are k proposals ending at every time step (potentially; at max), but only one LSTM hidden state for that time step. Hence, to get a more discriminative proposal representation, this LSTM hidden state (proposal state info) os fused (using attention) with video features (C3D sequences). Intuition: corresponding visual features will help discriminate overlapping features.
2. Hence, this fusion of encoded visual features + proposal states (which have past + future context + current info) is the input to an LSTM decoder. 
	1. Simple concatenation is not possible, since dim of visual features is different
	2. Mean pooling of visual feature is not taken as it does not explicity explore the relationship between an event and surrounding contexts.
	3. Hence: Temporal dynamic attention, i.e. a dynamic attention mechanism to fuse visual features and context vectors (proposal states). So visual features are attended wrt context vectors.
	4. Instead of simple concatenation of attended visual features and context features, a ***context gate*** is used to balance their relative contribution. The context gate explicitly measures contribution of surrounding context info (proposal state). Intent: network should learn how much context should be used when generating next word.
3. Joint ranking technique is used to select high-confidence proposal-caption pairs by taking proposal score and caption confidence.

![[Pasted image 20210729214025.png]]

## Training
Apparently, all parts involve LSTMs. Refer the end of 3.3.
Losses:
1. Proposal Loss: Weighted multi-label cross entropy
	- Lengths of all ground-truth labels are grouped into *k=128* clusters (anchors).
	- Each training example $v$ from $V$ is associated with a *k*-dimensional ground-truth label *y*, which has binary entries.
	- If temporal IoU of a proposal with ground truth is > 0.5, then 1, else 0.
	- Loss function: Weighted multi-label cross-entropy, weights determined by the number of positive and negative proposal samples.
	- Losses for forward and backward passes are calculated in the same way
	- These losses are added for training the forward and backward proposal modules.
	- Total proposal loss is obtained by averaging along time steps (does that mean for one video, the loss is average?), and for all videos
2. Captioning Loss
	- Only those proposals are given to Captioning module which have tIoU > 0.8 with the ground-truth proposals.
	- Captioning loss: sum of (negative log likelihood of (correct word in  a sentence with M words))
3. Total Loss = $\lambda * L_p + L_c$, where $\lambda$ is set simply to 0.5. (How does it balance though?)

## Results
The paper consists of comparisons with existing models as well as comparison of different combinations of the techniques outlined in the paper:
- (Bi-)SST
- Context vectors
- Event clip features
- Temporal Dynamic Attention
- Context Gating
- Joint Ranking

The best model is the combination of all of these.