
# Speaker Diarization Techniques for NPR Podcast Data

This repository contains implementations of speaker diarization techniques applied to NPR Podcast data. The techniques are implemented in two Jupyter Notebook files:

### 1. **mv3p.ipynb**: This notebook primarily focuses on speaker diarization using a Conditional Hidden Markov Model (CHMM).

The CHMM is described by:
$$\textbf{x}_{i+1} = A_{z_{i-1}} \textbf{x}_i \qquad \textbf{z}_{i} = B_{z_{i-1}} \textbf{x}_i. \qquad$$
Entries of transition matrix $A_{z_{i-1}}$ are $[a]_{ij} = P(X_i | X_j, Z_{i-1})$ and entries of emission matrix $B_{z_{i-1}}$ are $[b]_{ij} = P(Z_i | X_j, Z_{i-1})$. In a traditional HMM, the probability of $a_{ij}$ does not vary with $Z_{i-1}$ via the independence assumption of $P(Z_i | X_j) = P(Z_i | X_j, Z_{i-1})$. However, with CHMM, $a_{ij}$ would vary depending on the previous observation. Thus, we condition on $Z_{i-1}$ by introducing a new transition and emission matrix for every possible state in the observation space. 

In the context of speaker diarization with a state space of two speakers and observation space of $d$ words, the $2 \times 2$ transition matrix between speaker states varies depending on its previous word $Z_{i-1}$. For example, the transition matrices associated with previous words 'welcome' and 'really' would vary. The word "welcome" exhibits large off-diagonal entries in the transition matrix $A_{welcome}$, indicating a high probability of state change, whereas "really" demonstrates prominent diagonal entries in $A_{really}$, reflecting a greater likelihood of state persistence. Thus, we collect all $d$ of our conditional transition matrices and construct transition tensor $A_k^{ij} = \begin{bmatrix} A_1 & A_2 & \cdots & A_d \end{bmatrix}$ with a shape of $(d,2,2)$. The previous observation word $Z_{i-1}$ indexes each transition matrix and follows a similar process for our emission tensor $B_k^{ij} = \begin{bmatrix} B_1 & B_2 & \cdots & B_d \end{bmatrix}$ with a shape of $(d,d,2)$ as seen in Figure 1.

<p float="left">
  <img src="files_readme/HMM params.png" width="400" />
  <img src="files_readme/CHMM params.png" width="358" />
</p>

*Figure 1: A visual comparison between traditional HMM parameters and CHMM tensor parameters indexed by Z_{i-1}.*

Taking the transition and emission tensors corresponding to our CHHM model, we have a total of $4d + 2d^{2}$ parameters to estimate for our model. Given that $d = 10,107$, the number of parameters to estimate is beyond unreasonable for the Baum-Welch algorithm to estimate (roughly $204$ million parameters).

To make predictions under this new model, we modify the Viterbi algorithm for HMM to find the most probable sequence of hidden states for an entire episode. The Viterbi algorithm uses Bellman's principle to calculate the optimal path, using parameters from both transition and emission matrices at each time step. Our modified Viterbi algorithm follows the same procedure, only now (1) these matrix parameters are no longer temporally homogeneous, and (2) varying sets of parameters will be indexed according to the previous word $Z_{i-1}$ in our observation space. All other aspects of the algorithm remain the same, including the back-pointing procedure once the maximum $\eta_{ij}$ values have been identified.



### 2. **final.ipynb**: This notebook primarily focuses on speaker diarization using embedding and clustering techniques.

The clustering technique we employ is similar to other speaker diarization methods. First, we embed the data using OpenAI’s embedding algorithm. Then we use a clustering algorithm on the embedded data to predict who is speaking. We see the result of the embedding and clustering in Figure 2. 

<img src="files_readme/umap_embeddings.png" alt="UMAP Embeddings" width="500" />

*Figure 2: UMAP of the resulting embeddings. Clusters are shown by color.*

### 3. **misc_files**: This directory contains rough drafts used as working copies to obtain our results. 

### 4. **New_Phone_Who_This-Speaker Diarization Methods.pdf**: This PDF contains the write up of our methods and results. 



## Dataset

The speaker diarization techniques presented in these notebooks utilize the [NPR Media Dialog Transcripts dataset](https://www.kaggle.com/datasets/shuyangli94/interview-npr-media-dialog-transcripts). This dataset contains transcripts of interviews from NPR media dialogues.

