# Our settings
Here we describe a short description of the procedure we have used for post-training of Falcon-H1 models. We had two stages: SFT followed by DPO. We provide this as a reference and a possible starting point for the community to fine-tune Falcon-H1 models.

## SFT

Our SFT had a main 16k stage for 3 GigaTokens (GT), followed by 3GT of 128k long context stage. In the main 16k stage, we have used WSD learning rate schedule with exponential shape of the decay stage reducing the learning rate 8 times: $\eta_\mathrm{min}=\eta/8$. Then, the 128k stage continued with constant LR equal to the minimal WSD learning rate $\eta_\mathrm{min}$ of the 16k stage. Also, for our smallest 0.5B model, we did not perform the 128k stage due limited capacity of small models in processing long sequences. 

In terms of the data mix, we have various open-source SFT datasets as well as our internal ones, with Tulu3 serving as the basis of the mix with a 50% sampling weight. Importantly, after fixing the weight of each source in the mixture, we did not construct a single mixed dataset but instead sampled from each source independently. As a result, different data sources were repeated for different amounts of time (i.e. epochs) during the SFT stage. For example, Tulu3 with 50% weight had $\approx$ 1.7GT epoch size, resulting in $\approx3.5$ epochs during the whole 6GT duration of SFT. Tulu3 was the most repeated data source, with many of the other sources having  $\lesssim 2$ epochs, and hence little or no repetition at all. We hope that this information will be useful in making decisions on fine-tuning duration given the size of the fine-tuning dataset in a given use case.  

We summarize the main parameter of our SFT setting in the table below.

| Hyperparameter / Setting | Value / Details |
| --- | --- |
| Sequence Length | 16k for the main stage; 128k for long context stage |
| Batch size $b$ | 1 million tokens (MT) |
| Learning Rate $\eta$ | $128\times 10^{-6}$|
| AdamW parameters | $\beta_1=0.9$; $\beta_2=0.95$, no weight decay|
| Learning rate schedule | WSD: 50MT warmup, 1.5GT stable, 1.5GT decay |
| LR Decay | exponential schedule from $\eta$ to $\eta_\mathrm{min}=\eta/8$ |
| Long context stage | +3GT at then end of WSD with minimal LR $\eta_\mathrm{min}$ |
| # epochs for each data source | $\lesssim 3.5$ |

### Difference between model sizes

Up to this point, we have described the single setting for all the model sizes, from 0.5B to 34B (with the exception of no 128k stage for 0.5B). To a large extent, a single setting is possible due to the use of maximal-update-parametrization (μP) that tries to ensure that training dynamics stay roughly the same between different model sizes by scaling Hyperparameters with μP scaling rules. Our implementation of μP uses scaling multipliers directly in the forward pass of the model and therefore does not require scaling the learning rate across the model sizes. We plan to provide the script of how we scale different μP multipliers with the model's tensor shapes in the near future.  

One parameter that we have changed significantly across model sizes was batch size. In the ablations studies, we have noticed that different batch sizes in the range from 0.25MT to 4MT have little impact on the SFT results, with smaller batch sizes being only slightly better in some scenarios. Therefore, we were mostly choosing batch size for each model to optimally distribute compute nodes across different model sizes to ensure the timely completion of the SFT stage. For example, longcontext 128k stage was slow due to context parallelism and therefore required larger batch sizes to allow for larger data parallelism (DP) degrees. 

However, different batch sizes have different optimal learning rates. We minimize the effect of the shifting of the optimal LR with *Adam batch scaling* that keeps $\frac{\eta}{\sqrt{b}}$ constant when changing batch size. Specifically, at each batch size $b$ we were setting the maximal LR of the WSD schedule according to 
$$
\eta(b) = \eta_\mathrm{ref}\sqrt{\frac{b}{b_\mathrm{ref}}},
$$
where $ \eta_\mathrm{ref}=128\times10^{-6}$ and $b_\mathrm{ref}=$ 1MT are the values we previously mentioned in the table. This scaling captures most of the optimal LR shift and removes the need to re-tune LR for each new batch size. We are expecting that with Adam scaling, researchers and developers finetuning Falcon-H1 models can use our hyperparameters as a good starting point, most importantly learning rate, while keeping the batch size more suitable for their scenario.   

## DPO

We use a fork of AllenAI open-instruct repo. Similarly to the SFT stage, we constructed our data mix based on Tulu3 plus a few additional open-source datasets. We used standard DPO loss (`dpo_loss_type: dpo_norm`) and the same hyperparameters outlined below

| Hyperparameter / Setting | Value / Details |
| --- | --- |
| Batch size| 256 |
| Learning rate | $5\times 10^{-6}$ |
| Learning rate schedule | Linear to zero over 2 epochs, warmup ratio 0.1|
| DPO loss parameters | $\beta=5$ |
| AdamW parameters| pytorch default ($\beta_1=0.9$, $\beta_2=0.999$)|

Importantly, for our final checkpoints we were stopping the training not at 2 epochs, corresponding to the end of the linear schedule, but at approriximately 1 epoch. We have found this strategy to work better than taking the last checkpoint at 2 epochs or using linear scheduler that finishes at 1 epoch.  



