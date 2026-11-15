# literature review evidence: [Uncertainty Estimates and Multi-Hypotheses Networks for Optical Flow](../../Papers/1802.07095.pdf)

Ilg et al., ECCV 2018 · bib key `ilg2018uncertainty` · report §2.2

## ai-generated summary review

### what the paper does

It compares ways of making an optical flow network (FlowNetC) output a per-pixel uncertainty: empirical ensembles (MC dropout, bootstrapped ensembles, SGDR snapshot ensembles), a single network that predicts a variance, and ensembles of such predictive networks. It also proposes FlowNetH, a multi-hypotheses network trained with a Winner-Takes-All loss plus a merging network, which gives uncertainty in one forward pass.

- p. 1, Abstract, lines 5-7: "For the first time we compare several strategies and techniques to estimate uncertainty"
- p. 2, §1, ¶2 (first full paragraph under Fig. 1), lines 4-10: a study of empirical ensembles, "predictive models, and predictive ensembles", implemented with MC dropout, Bootstrapped Ensembles and snapshot ensembles
- p. 2, §1, ¶4 (starts "As a second contribution"), lines 1-6: the multi-headed network with "the Winner-Takes-All (WTA) loss" and a stacked second network that merges the hypotheses
- p. 12, §5.3, paragraph "Predictive Uncertainty Estimation", lines 2-4: a single predictive FlowNetC "yields much better uncertainty estimates than any empirical ensemble in terms of AUSE"
- p. 12, §5.3, paragraph "Ensemble Types", lines 1-2: "the commonly used dropout [9] technique performs worst in terms of EPE and AUSE"

### the one result or idea that matters to us

The paper does not originate the sparsification plot: it says such plots are already commonly used and cites four earlier optical flow papers. What it introduces is the Sparsification Error, the difference between a method's sparsification curve and its own oracle curve (ranking by true error), and the single number AUSE, the Area Under the Sparsification Error curve, so that methods with different oracles can be compared.

- p. 10, §5.2, paragraph "Sparsification Plots", lines 1-2: "so-called sparsification plots, which are commonly used for this purpose [1,33,19,20]"
- p. 10, §5.2, paragraph "Sparsification Plots", lines 7-8: ranking by the true error; "We refer to this curve as Oracle Sparsification"
- p. 10, §5.2, paragraph "Sparsification Error", lines 1-4: "we introduce a measure, which we call Sparsification Error", defined as "the difference between the sparsification and its oracle"
- p. 10, §5.2, paragraph "Sparsification Error", lines 7-8: "the Area Under the Sparsification Error curve (AUSE )"
- pp. 18-19, Supplementary §3 "Sparsification Plots", ¶1 (runs across the page break), last sentence, and the paragraph "Image-wise sparsification", lines 1-3: "the errors are normalized to 1"; normalization and sparsification are done per image and the plots of all images are then averaged, which the main paper follows. The y-axis of Fig. 3 (p. 10) is labelled "Average EPE (Normalized)".

### how it relates to our design

Our evaluation metric is this paper's AUSE, reached through Poggi et al., who take the sparsification plot from this paper and add AURG. The paper's related work also describes the design our head follows: a post-hoc binary classifier that predicts whether a pixel's error exceeds a threshold and uses the probability as the uncertainty (Mac Aodha et al., for optical flow).

- p. 10, §5.2, paragraph "Sparsification Error", lines 1-8: definition of Sparsification Error and AUSE (as above)
- Poggi et al. ([2005.06209](../../Papers/2005.06209.pdf)) p. 6, §4.1, paragraph "Uncertainty metrics", right column, lines 1-3: "we use sparsification plots as in [27]"; reference [27] on p. 10 is this paper
- p. 3, §2, ¶2 (starts "Post-hoc methods", under "Confidence measures for optical flow"), lines 5-8: Aodha et al. [1] "trained a binary classifier to predict whether the endpoint error of each pixel is bigger or smaller than a certain threshold"
- [literature-review.tex:31](../latex/sections/literature-review.tex:31): §2.2 opening; "sparsification curves and AUSE come from this literature", and we are closest to the post-hoc methods because the backbone is frozen
- [literature-review.tex:39](../latex/sections/literature-review.tex:39): the \todo stub for this paper; [literature-review.tex:41](../latex/sections/literature-review.tex:41): our AUSE and AURG definitions via Poggi et al., and the note that a classifier trained on the prediction error "is what our head is"

### notes for the writer

- The stub at [literature-review.tex:39](../latex/sections/literature-review.tex:39) calls this paper "the origin of the sparsification plot". The paper does not support that: p. 10, §5.2 says the plots "are commonly used for this purpose [1,33,19,20]" (Mac Aodha et al. 2013, Wannenwetsch et al. 2017, Kondermann et al. 2008, Kybic and Nieuwenhuis 2011), and the supplement (p. 18, §3) says "Already popular in literature [4,19,20,1]". Safe wording: the paper introduces the sparsification error and AUSE; the plot itself is older.
- The stub says Poggi et al. "adopt as AUSE" the area measure, which could read as if Poggi coined the name. The name and acronym AUSE are already in this paper (p. 10, §5.2). Poggi et al. reuse it and add AURG; their text cites this paper for the sparsification plots, not explicitly for AUSE.
- Normalisation caveat: the main body never states how the curves are normalised; only the Fig. 3 axis label and the supplement (pp. 18-19) do, and the supplement does not give a formula ("the errors are normalized to 1"). The error is endpoint error for optical flow, per pixel and per image. Do not claim our grid-cell AbsRel curves follow the paper's exact procedure; literature-review.tex:41 already says the area computation is our own choice.

## post-review discussion with writer
