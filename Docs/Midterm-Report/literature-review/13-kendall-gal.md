# literature review evidence: [What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?](../../Papers/1703.04977.pdf)

Kendall and Gal, NeurIPS 2017 (printed as NIPS 2017) · bib key `kendall2017uncertainties` · report §2.2

## ai-generated summary review

### what the paper does

It separates two kinds of predictive uncertainty, aleatoric (noise inherent in the observations, not reducible with more data) and epistemic (uncertainty in the model, reducible with more data), and builds one network that estimates both for per-pixel depth regression and semantic segmentation. Aleatoric uncertainty is a per-pixel variance the network outputs next to its prediction; epistemic uncertainty comes from Monte Carlo dropout sampling.

- p. 1, Abstract, lines 1-4: the two definitions; "Aleatoric uncertainty" on line 1 and "captures noise inherent in the observations" on line 2; epistemic is "uncertainty which can be explained" away given enough data (lines 3-4)
- p. 1, Abstract, lines 7-10: the framework and the two tasks; "with per-pixel semantic segmentation and depth regression tasks" (line 10)
- p. 2, §1, first paragraph on the page (continued from p. 1), lines 3-5: aleatoric splits into homoscedastic and heteroscedastic; "Heteroscedastic uncertainty depends on" the inputs to the model
- p. 6, §4, ¶3, lines 1-4: how each is implemented; "We model epistemic uncertainty using Monte Carlo dropout", "we use 50 Monte Carlo dropout samples", and aleatoric "with MAP inference using loss functions (8)"
- p. 7, Table 2b (NYUv2 Depth), "This work" rows, rel column: baseline 0.117, + aleatoric 0.112, + epistemic 0.114, + both 0.110

### the one result or idea that matters to us

Heteroscedastic regression with a learned variance: the network's head is split to output a depth and a variance per pixel, and the variance is learned from the ordinary regression labels through the loss in Eq. 7-8, with no uncertainty labels. The paper then shows what each kind captures: aleatoric stays roughly constant with more data and does not rise on out-of-distribution inputs, while epistemic shrinks with more data and rises on out-of-distribution inputs.

- p. 5, §3.1, the sentence after Eq. 6, line 2: one network "with its head split to predict both" ŷ and σ̂²
- p. 5, §3.1, Eq. 7 and Eq. 8: the loss, residual weighted by 1/σ̂² plus a log σ̂² term; Eq. 8 is the same with the network predicting "the log variance"
- p. 5, §3.1, paragraph after Eq. 7 ("This loss consists of two components"), lines 3-5: no uncertainty labels needed, only "supervise the learning of the regression task"; the variance is learned "implicitly from the loss function"
- p. 9, §5.2, the two numbered results and ¶1 below them: "Aleatoric uncertainty cannot be explained away with more data" and it "does not increase for out-of-data examples"
- p. 9, Table 3a (regression), columns Aleatoric variance / Epistemic variance: trained on Make3D / 4, Make3D / 2, Make3D and tested on Make3D gives aleatoric 0.506, 0.521, 0.485 and epistemic 7.73, 4.38, 2.78; trained on Make3D and tested on NYUv2 gives epistemic 4.87

### how it relates to our design

Both of the paper's estimators are part of the depth network and are trained with it (a variance output on the depth head, dropout in the layers), which is the "changes the depth model" family our report sets itself apart from because our backbone is frozen. Our head does not output a variance of either kind: it is a separate network trained with binary cross-entropy on lidar-derived labels to output the probability that the depth error exceeds a threshold.

- p. 5, §3.1, the sentence after Eq. 6, line 2: the variance is a second output of the same network that predicts depth, "with its head split to predict both"
- p. 6, §4, ¶3, lines 2-3: epistemic uncertainty needs dropout inside the depth network; "places dropout with p =" 0.2 after each convolutional layer
- p. 9, §5.3, ¶1, lines 2-6: cost; "The aleatoric uncertainty models add negligible compute", while dropout sampling "often results in a 50× slow-down for 50 Monte Carlo samples"
- p. 8, §5.1, ¶1, lines 2-3: the paper evaluates uncertainty by "removing pixels with uncertainty larger than various" percentile thresholds and plotting the error of the rest (Figure 2), the same idea as the sparsification protocol in [literature-review.tex:41](../latex/sections/literature-review.tex:41)
- [literature-review.tex:31](../latex/sections/literature-review.tex:31): "Most work on uncertainty for depth changes the depth model: it adds a variance output, trains several models, or samples a stochastic one. Our setting forbids that, since the backbone is frozen"
- [literature-review.tex:33](../latex/sections/literature-review.tex:33): the `\todo{}` stub for this paper, ending "Our head is a classifier on the error, which sidesteps the choice."
- [introduction.tex:9](../latex/sections/introduction.tex:9) and [progress.tex:36](../latex/sections/progress.tex:36): our head outputs "the probability that the depth there is wrong by more than a threshold"; "The loss is binary cross-entropy over labeled cells only"

### notes for the writer

- "Sidesteps the choice" is our framing, not the paper's. Absence claim, not quotable: the paper never discusses predicting a thresholded error, or any uncertainty estimator trained separately from the predictor, so it cannot be cited for the claim that an error classifier avoids the aleatoric/epistemic question. The nearest supported statement is p. 8, §5.1, ¶1, lines 7-10: with only one kind modeled, each "ranks pixel confidence similarly to" the other, and the one modeled "attempts to compensate for the lack of the alternative uncertainty". A safer wording is that our head does not try to separate the two.
- Caveat against overclaiming: the paper says a network trained by MAP inference to output uncertainty "does not capture epistemic model uncertainty" (p. 4, §2.2, paragraph after Eq. 5, lines 3-4), and that out-of-data examples "cannot be identified with aleatoric uncertainty alone" (p. 2, §1, paragraph "In this paper we make the observation", lines 5-6). The paper makes these statements about its own heteroscedastic model, not about a head like ours, but they are a reason not to write that our head handles both kinds.
- Do not cite §3.3 ("Heteroscedastic Uncertainty in Classification Tasks", pp. 5-6) as related to our classifier. It puts Gaussian noise on the logits of a semantic segmentation network (Eq. 10-12); it is not a classifier on the error. Also, the experiments use a Laplacian prior, so the trained loss applies L1 to the residuals, not the Gaussian form of Eq. 7-8 (p. 6, §4, ¶3, lines 5-7: "a Laplacian prior, as opposed to the Gaussian prior").

## post-review discussion with writer
