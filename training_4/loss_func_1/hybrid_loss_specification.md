# Hybrid Autoregressive and Distance-Weighted Denoising Loss

Status: proposed experimental objective, not a validated result or a claim of novelty.

## 1. Goal and scope

Train one model to retain ordinary next-token prediction while learning to reconstruct multiple masked future tokens. Near-prefix errors receive more emphasis, distant errors retain a positive contribution, and each example's token coefficients sum to one. This specifies a masked-token language-model objective, not continuous embedding diffusion or reinforcement learning.

The primary proposal omits a separate diffusion loss. Its denoising branch still uses corrupted inputs sampled at multiple noise levels. A loss alone does not specify the inference sampler or guarantee fast or coherent generation.

## 2. Complete primary objective

$$
L(\theta;s)=\alpha L_{\mathrm{AR}}(\theta)+(1-\alpha)L_{\mathrm{distance}}(\theta;s).
$$

$$
L_{\mathrm{AR}}=\mathbb E_{x\sim\mathcal D}\left[-\frac1T\sum_{i=1}^T\log p_\theta^{\mathrm{AR}}(x_i\mid x_{<i})\right].
$$

$$
L_{\mathrm{distance}}=\mathbb E_{x,c,\tau,\tilde x}\left[\sum_{i\in M}q_i(s)\ell_i\right],
\qquad
\ell_i=-\log p_\theta^{\mathrm{den}}(x_i\mid x_{\le c},\tilde x_{c+1:c+H},\tau).
$$

$$
d_i=i-c,\qquad
q_i(s)=\frac{\rho}{|M|}+(1-\rho)\frac{2^{-(d_i-1)/h(s)}}{\sum_{j\in M}2^{-(d_j-1)/h(s)}}.
$$

Equivalently, substituting all definitions:

$$
\boxed{L(\theta;s)=\alpha\mathbb E_x\left[-\frac1T\sum_{i=1}^T\log p_\theta^{\mathrm{AR}}(x_i\mid x_{<i})\right]+(1-\alpha)\mathbb E_{x,c,\tau,\tilde x}\left[\sum_{i\in M}\left(\frac{\rho}{|M|}+(1-\rho)\frac{2^{-(i-c-1)/h(s)}}{\sum_{j\in M}2^{-(j-c-1)/h(s)}}\right)\left(-\log p_\theta^{\mathrm{den}}(x_i\mid x_{\le c},\tilde x_{c+1:c+H},\tau)\right)\right].}
$$

All logarithms are natural; losses are in nats. AR and den superscripts indicate prediction modes with shared trainable parameters, not necessarily separate models or heads.

## 3. Every symbol

| Symbol | Definition |
|---|---|
| $\theta$ | All trainable model parameters shared by the two modes |
| $L$ | Combined training objective |
| $\mathcal D$ | Training distribution of tokenized lyric sequences |
| $x$ | Clean sequence $(x_1,\ldots,x_T)$ |
| $T$ | Number of valid AR target tokens, excluding padding; first token is conditioned on a beginning-of-sequence token |
| $x_i$ | Ground-truth token at position $i$ |
| $i,j$ | Token-position indices; $j$ is a summation index |
| $x_{<i}$ | Clean tokens before position $i$ |
| $c$ | Last position of the clean conditioning prefix |
| $x_{\le c}$ | Clean prefix through position $c$; empty when $c=0$ |
| $H$ | Actual target-block length, with $1\le H\le T-c$ |
| $\tau$ | Sampled corruption/noise level; distinct from training progress |
| $\tilde x$ | Corrupted version of the target block |
| $M$ | Nonempty set of masked target positions within $\{c+1,\ldots,c+H\}$ |
| $|M|$ | Number of masked targets |
| $d_i$ | Absolute positional distance from the clean prefix boundary, $i-c$ |
| $p_\theta^{\mathrm{AR}}$ | Causal next-token predictive distribution |
| $p_\theta^{\mathrm{den}}$ | Reconstruction distribution given prefix and corrupted block |
| $\ell_i$ | Cross-entropy for one masked target |
| $q_i(s)$ | Normalized coefficient assigned to masked target $i$ |
| $\alpha$ | AR mixture coefficient |
| $\rho$ | Fraction of denoising coefficient mass allocated uniformly |
| $h(s)$ | Distance half-life, measured in token positions |
| $s$ | Training progress between zero and one |
| $\mathbb E$ | Expectation over data and the explicitly sampled training setup |

The expectation for denoising includes the chosen prefix distribution, block-length policy, noise-level distribution, and random masking. These choices are part of the experiment and must be recorded.

## 4. Why each term exists

### Autoregressive term

Each target is predicted from its correct preceding tokens under causal attention. Averaging by valid sequence length avoids a purely length-driven increase in loss. This keeps causal prediction in the optimization objective, but does not freeze the pretrained model or guarantee preservation of all its abilities.

### Distance decay

The unnormalized coefficient halves every $h$ positions. If two masked targets differ in distance by $h$, the exponential part of the nearer target's coefficient is twice as large. The complete coefficient ratio also depends on the uniform floor.

Distance always refers to the fixed clean-prefix boundary. It does not mean distance from the nearest visible token, nor rank among masked positions. With partially visible future tokens, proximity to the prefix is a chosen priority, not a guaranteed measure of uncertainty.

### Uniform contribution

The uniform component gives every masked target at least $\rho/|M|$ of the denoising coefficient mass. This lower bound depends on the number of targets; it is not a fixed absolute floor across block lengths.

### Normalization

For every valid example, $q_i>0$ and $\sum_{i\in M}q_i=1$. If all target cross-entropies equal $C$, then the example's distance loss equals $C$. Here $C$ denotes any common nonnegative cross-entropy value.

Normalization keeps total coefficient mass comparable. It does not ensure identical loss values, gradient norms, task difficulty, or effective sample sizes. A longer sequence or more corrupted input can still be harder.

## 5. Hyperparameters and design choices

| Setting | Domain / units | Role and effect |
|---|---|---|
| $\alpha$ | $[0,1]$ | Higher values emphasize AR; one means AR-only, zero means denoising-only |
| $\rho$ | $[0,1]$; use strictly positive for a floor | Higher values distribute more denoising weight uniformly; one removes distance preference |
| $h$ | Positive token count | Smaller values concentrate on nearest masked targets; larger values approach uniform weighting |
| $h_{\mathrm{start}},h_{\mathrm{end}}$ | Positive token counts | Optional initial/final half-lives for a curriculum |
| $S$ | Positive integer optimizer steps | Planned training length used to define progress |
| $H_{\max}$ | Positive integer tokens | Maximum target-block length; use actual $H=\min(H_{\max},T-c)$ |
| Prefix sampling policy | Distribution over $c\in\{0,\ldots,T-1\}$ | Determines which positions become near-prefix targets |
| Noise-level sampling policy | Distribution over $\tau$ | Determines exposure to easy versus heavily corrupted inputs |
| $m(\tau)$ | Masking probability in $[0,1]$ | Maps noise level to corruption severity |
| Empty-mask policy | Resample, explicitly condition, or documented alternative | Avoids division by zero; changes the sampled corruption distribution |
| $B$ | Positive integer examples | Batch size for averaging per-example losses |
| $\eta$ | Positive learning rate | Optimizer update scale; distinct from objective coefficients |
| Optimizer / clipping settings | Implementation choices | Affect update stability; not defined by this objective |
| $\beta$ | Optional, $[0,1]$ | Distance fraction within the denoising branch when retaining a third loss |

No numerical setting has been validated. For an interpretable starting configuration, one could test $\alpha=0.7$, $\rho=0.1$, and a fixed half-life chosen relative to the target-block length. The $h=1$ example below illustrates the arithmetic and is not a recommendation for long blocks.

Keep coefficients externally specified initially. Learning them solely by minimizing this same loss can encourage the model to reduce emphasis on difficult tasks rather than learn them.

### Optional training curriculum

Let $u$ be the number of completed optimizer steps and define $s=\min(u/S,1)$. A linear half-life schedule is

$$
h(s)=h_{\mathrm{start}}+s(h_{\mathrm{end}}-h_{\mathrm{start}}).
$$

With $h_{\mathrm{end}}>h_{\mathrm{start}}$, emphasis broadens over training. This is a proposed curriculum, not an established improvement. Start with a constant half-life to isolate the effect of distance weighting. Keep $\alpha$ and $\rho$ fixed in that first comparison.

## 6. Data and attention requirements

1. Sample a clean tokenized sequence and compute its AR loss with shifted targets and strictly causal attention.
2. Sample a clean-prefix boundary and target block.
3. Sample $\tau$ and mask target-block tokens using the documented corruption policy. A simple possible policy is independent masking with probability $m(\tau)=\tau$, followed by resampling masks when none are selected. This conditions the mask distribution on being nonempty and must not be mistaken for an unchanged diffusion-likelihood estimator.
4. Compute denoising logits from the prefix and corrupted block. Masked target tokens must not be exposed to their own predictions. Unmasked block tokens may supply context if the intended denoising attention permits it.
5. Compute unreduced CE only at masked, non-padding targets; normalize distance coefficients over exactly those positions.
6. Average per-example losses over the batch and combine the two branches.

The batch estimate is $\widehat L=B^{-1}\sum_{b=1}^B[\alpha L_{\mathrm{AR},b}+(1-\alpha)L_{\mathrm{distance},b}]$, where $b$ indexes examples and the hat denotes a sampled estimate. Each example receives equal weight; this differs from pooling every valid token across the batch before averaging.

Use separate forward passes or another correctly isolated implementation for the two attention modes. Changing coefficients alone cannot turn a causal model into a bidirectional denoiser. The architecture, positional alignment, mask-token representation, and inference sampler must support the intended task. Explicit noise-level conditioning is optional architecturally; the notation includes it as available information.

## 7. Effect on model parameters

For fixed externally supplied coefficients:

$$
\nabla_\theta L=\alpha\nabla_\theta L_{\mathrm{AR}}+(1-\alpha)\mathbb E\left[\sum_{i\in M}q_i(s)\nabla_\theta\ell_i\right].
$$

Here $\nabla_\theta$ denotes the gradient with respect to trainable parameters. A plain gradient-descent update is

$$
\theta_{\mathrm{new}}=\theta_{\mathrm{old}}-\eta\nabla_\theta L.
$$

Adaptive optimizers transform this gradient further. Loss coefficients multiply gradient contributions, not the parameter values directly.

Let $z_{i,k}$ be the denoising logit for vocabulary entry $k$, $p_{i,k}$ its softmax probability, and $\mathbf1(k=x_i)$ an indicator equal to one when $k$ is the target token and zero otherwise. For one example's denoising contribution:

$$
\frac{\partial L}{\partial z_{i,k}}=(1-\alpha)q_i(s)[p_{i,k}-\mathbf1(k=x_i)].
$$

The batch version adds the factor $1/B$. For equal prediction errors, nearby targets receive larger logit corrections. Parameter gradients also depend on the model Jacobian, so this does not guarantee larger parameter changes for every nearby token.

An AR coefficient of 0.7 is not a guarantee of 70% of the actual update. Gradients can differ in magnitude and can reinforce or oppose each other. Normalized positive coefficients prevent growth merely from summing extra targets, but do not guarantee freedom from exploding gradients or catastrophic forgetting.

## 8. Numerical example and limiting cases

Take four masked targets at distances 1, 2, 3, and 4, with $h=1$, $\rho=0.1$, and $\alpha=0.7$.

| Distance | Unnormalized exponential | Normalized $q_i$ | Coefficient in total loss |
|---:|---:|---:|---:|
| 1 | 1 | 0.505 | 0.1515 |
| 2 | 0.5 | 0.265 | 0.0795 |
| 3 | 0.25 | 0.145 | 0.0435 |
| 4 | 0.125 | 0.085 | 0.0255 |

The $q_i$ values sum to 1; their total-loss coefficients sum to 0.3. The AR mean receives coefficient 0.7.

- $\rho=1$: uniform masked-token mean, regardless of half-life.
- $\rho=0$: pure exponential weighting, with no explicit uniform floor.
- $h\to\infty$: uniform weighting over masked targets.
- $h\to0^+$: the exponential share concentrates on the nearest masked target; the uniform share remains.
- $|M|=1$: that target receives coefficient 1, so distance weighting has no effect.
- $|M|=0$: the formula is undefined; apply the documented sampling policy.

## 9. Optional third diffusion term

The general extension is

$$
L=\alpha L_{\mathrm{AR}}+(1-\alpha)[(1-\beta)L_{\mathrm{diffusion}}+\beta L_{\mathrm{distance}}].
$$

The outer coefficients are $\alpha$, $(1-\alpha)(1-\beta)$, and $(1-\alpha)\beta$, which sum to one. $\beta=1$ gives the primary two-term proposal. $\beta=0$ gives AR plus the original diffusion objective.

There is no universal diffusion loss independent of the diffusion formulation. Preserve the implementation's exact noise distribution, time weights, reduction, and any boundary terms when calling it the original diffusion objective.

If the baseline is specifically the uniform masked-token mean on the same input distribution,

$$
L_{\mathrm{diffusion}}^{\mathrm{mean}}=\mathbb E\left[\frac1{|M|}\sum_{i\in M}\ell_i\right],
$$

then the mixture is algebraically equivalent to the same distance formula with

$$
\rho_{\mathrm{effective}}=1-\beta+\beta\rho.
$$

Here $\rho_{\mathrm{effective}}$ is the resulting uniform coefficient mass. Thus $\beta$ and $\rho$ are redundant controls under these precise assumptions.

For an actual diffusion objective with noise-dependent weights or different sampling, this equivalence need not hold. Replacing it with the normalized distance objective changes training and does not automatically retain its likelihood-bound interpretation. If retaining it, coefficient sums alone do not make its scale comparable to the two per-example means.

## 10. Implementation and evaluation notes

Compute the exponential distribution stably with softmax over masked positions using scores $-(d_i-1)\ln(2)/h(s)$. Softmax subtracts the maximum score internally and avoids all raw exponentials underflowing for distant blocks. Combine this distribution with the uniform floor afterward.

Exclude padding from all denominators and target sets. Compute weights separately for every example. Log both component losses and their weighted contributions. Inspect gradient norms and, when useful, gradient alignment; scalar loss values alone do not reveal interference.

Compare AR-only, AR plus uniform denoising, and AR plus distance-weighted denoising under matched data and compute budgets. If an existing diffusion objective is available, retain it as an additional baseline. Measure held-out causal CE, reconstruction CE by distance and noise level, and lyric quality under the actual intended sampler. Track generation latency separately: the loss itself does not provide a speed guarantee.

## 11. Provenance and unresolved choices

Cross-entropy, convex loss mixtures, masked denoising, and exponential weighting are established ingredients. This document assembles them into the proposed project objective; it does not establish novelty or empirical effectiveness. The target-block length, masking policy, attention configuration, inference sampler, and numerical hyperparameters remain experiment choices rather than previously agreed facts.

For background on masked diffusion objectives and their likelihood interpretation, see Sahoo et al., *Simple and Effective Masked Diffusion Language Models* (2024): https://arxiv.org/abs/2406.07524. This reference supports the masked-diffusion background, not a claim that the exact distance-weighted hybrid above comes from that paper.
