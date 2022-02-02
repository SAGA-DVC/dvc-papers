# End-to-end Dense Video Captioning with Parallel Decoding

## Limitations of previous work identified

### Two-stage design
* Localize then describe
* In some cases, cannot train end-to-end due to two stages
* Since cannot be trained end-to-end, mutual promotion of the two subtasks (temporal action localization and captioning) is limited;  captioning is considered a downstream task
* Since captioning is a downstream task, captioning quality relies on proposal generation performance
### Unreliable event number estimation
* Choice of the number of events: manual thresholds of confidence scores, top event selection methods, non-maximum suppression (NMS); all these are heuristic.
    * These methods introduce a lot of design issues, assumptions and hyperparameters. They are "*hand-crafted components*"
* SDVC's Event Sequence Generation Network generates the final proposals, but it is recurrent (hence not suitable for long videos) and cannot be trained end-to-end
* Unreliable event number estimation causes either:
    * Missing information in captions due to under-estimation
    * Redundant captions due to over-estimation
### Design of proposal generators
Most proposal generators use anchors and/or post-processing of events. Again, this brings in hyperparameters.

## What the proposed solution offers
Models the task as a set prediction problem, where the set to predict is a set of:
$$ \{ t_j^s, t_j^e, S_j \} $$
where $t_j^s$ and $t_j^e$ are the start and end times of the event, and $S_j$ is its caption.
This set is of size $N_{set}$, which is predicted by the **event counter**.

### Localization and caption generation in parallel
* Simpler pipeline; single stage; no "*hand-crafted components*"
* Ability to train end-to-end
* Mutual promotion of two tasks
### Predicting number of events
An event counter predicts the number of events in the video; it learns to do so, based on video understanding instead of heuristics. Hence no NMS or other methods required. This is claimed to be a more reliable event number estimation.

### Performance
* Locallization at par with SOTA
* Captioning quality *better* than SOTA

## Architecture
![PDVC Architecture](assets/Pasted%20image%2020220202001005.png)
### Feature Encoding
* Features extracted from videos using C3D (and TSN for comparison)
* $L$ temporal convolutional layers (convolutions over temporal dimension) to get **multi-scale features**, from $T$ to $T/2^L$ 
* Multi-scale frame features with positional encoding are given to the **encoder of deformable transformer** 
    * MSDAtt gives a context vector as output, termed $\{\mathbf{f}_l\}_{l=1}^L$

### Parallel Decoding
* Consists of the **deformable transformer decoder** and the three prediction heads - event counter, localization head and caption head
* MSDAtt also used here, in place of cross-attention; self-attention is unchanged
* Decoder layers query event level features from $\{\mathbf{f}_l\}_{l=1}^L$ conditioned on $N$ learnable embeddings (event query $q_j$) and corresponding scalar reference point $p_j$
    * Event query $q_j$  is an initial guess of event features
    * Reference point $p_j$ is an initial guess of center point of event
    * $q_j$ and $p_j$ are refined at each decoder layer. Final: $\tilde{q_j}$ and $\tilde{p_j}$*
    * $p_j$ is predicted by linear projection (matrix multiplication) and sigmoid activation of $q_j$
* The output (representation) from decoder layers is given to the three heads directly and simultaneously (in parallel).

### Localization Head
* Performs box prediction: center and length offset, w.r.t. reference point
* Performs binary classification: outputs foreground confidence distribution of each event query
* For both these tasks, uses MLP
* Output: $\{ t_j^s, t_j^e, c_j^{loc} \}$, where $c_j^{loc}$ is the localization confidence (foreground confidence)

### Captioning Head
Two variants proposed:
* lightweight: vanilla LSTM
* standard: LSTM using deformable soft attention (DSA)
#### Lightweight captioning head
* Vanilla LSTM
* Feed $\tilde{q}_j$ at each time step
* $h_{jt}$ (hidden state for $j^{th}$ event query at step (word) t) is passed through a FC layer with softmax activation giving output as next word distribution
* In this variant, only event-level features $\tilde{q_j}$ used; cues from the *combination* of caption words and event features are not used

#### Using Deformable Soft Attention (DSA)
* Soft Attention (SA): dynamically determines the importance of each frame when generating next word; used in DVC
* However, existing DVC methods are able to use this easily by restricting soft attention area to event boundaries
* In PDVC, we don't have event boundaries since captioning is in parallel with proposal generation. If we don't limit soft attention, optimization will be very slow.
* Proposed: Deformable Soft Attention (DSA). It enforces soft attention weights to focus on a small part around the reference points
    * When generating $t^{th}$ word $w_{jt}$:
        * Generate $K$ sampling points from each of the frame features $\{\mathbf{f}_l\}_{l=1}^L$ conditioned on $h_{jt}$ and $\tilde{q_j}$.
        * i.e., $K \times L$ sampling points are the keys and values, while $[h_{jt}, \tilde{q_j}]$ is the query in soft attention
        * Output of DSA: $z_{jt}$
* $[z_{jt}, \tilde{q_j}, w_{j,t-1}]$ given to LSTM as input
* FC + softmax for next word
* $S_j = \{w_{j1}, w_{j2}, ..., w_{jM_j}\}$ is the predicted sentence, of length $M_j$

### Event Counter
* Max-pooling layer to compress event queries $\tilde{q_j}$ into global feature vector
* FC layer to predict vector $r_{len}$
* $N_{set} = argmax(r_{len})$

Final output: select top $N_{set}$ events from $N$ event queries, using confidence score:
$$c_j = c_j^{loc} + \mu \frac{1}{M_j^\gamma} \sum_{t=1}^{M_j}{log(c_{jt}^{cap})}$$
Here, $\mu$ is the balance factor between localization and captioning confidence, and $\gamma$ is a modulation factor to rectify influence of caption length.


## Set Prediction Loss
* Hungarian Algorithm to match predicted events with ground truths to find best bipartite (the two sets are predicted events and ground truth events) matching results
* Matching cost:
$$C = \alpha_{giou}L_{giou} + \alpha_{cls}L_{cls}$$
* Set prediction loss:
$$L = \beta_{giou}L_{giou} + \beta_{cls}L_{cls} + \beta_{ec}L_{ec} + \beta_{cap}L_{cap}$$
Here, 
* $L_{giou}$: Generalized IoU between predicted temporal segments and ground truth segments
* $L_{cls}$: focal loss between predicted classification score and ground truth labels
* $L_{ec}$: cross-entropy loss between predicted event count distribution and ground truth
* $L_{cap}$: cross-entropy loss between predicted word distribution and ground truth (normalized by caption length)

## Multi-scale Deformable Attention
Refer:
* [Deformable DETR: Deformable Transformers for End-to-End Object Detection](https://arxiv.org/abs/2010.04159)
* [OpenReview](https://openreview.net/forum?id=gZ9hCDWe6ke)
Goal: To mitigate problems:
* Slow convergence: attention is initially uniformly distributed; for performance, it needs to be sparse
* Attention has quadratic complexity with spatial size; processing high resolution or multi-scale feature maps becomes difficult
![MSDAtt](assets/Pasted%20image%2020220202214416.png)
    * It attends to a sparse set of sampling points around certain reference point
    * Acting as a pre-filter for prominent key elements out of all feature map vectors
    * This is inspired by deformable convolutions
    * Sparsity of attention is data-dependent, and is learnt; contrasting to pre-defined sparse attention
![Deformable Attention aggregation](assets/Pasted%20image%2020220202215035.png)
![Multi-scale deformable attention aggregation](assets/Pasted%20image%2020220202215059.png)
* Ambiguity of MSDAtt actually being an attention mechanism, since attention weight $A$ does not involve a similarity score between query and keys: https://openreview.net/forum?id=gZ9hCDWe6ke&noteId=x1VT5henOtF*

