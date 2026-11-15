# literature review evidence: [Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning](../../Papers/1506.02142.pdf)

Gal and Ghahramani, ICML 2016 · bib key `gal2016dropout` · report §2.2

## ai-generated summary review

### what the paper does

It shows that training a neural network with dropout is approximate Bayesian inference in a deep Gaussian process, and from that derives "MC dropout": keep dropout switched on at test time, run T stochastic forward passes, and use their mean as the prediction and their spread as the model uncertainty. It tests this on small regression datasets, MNIST classification and a reinforcement-learning task, not on depth estimation.

- p. 1, Abstract, lines 8-15 (left column): the framework and what it yields; "new theoretical framework casting dropout", then "tools to model uncertainty with dropout NNs"
- p. 2, §3, ¶1, lines 1-4 (right column): the equivalence holds for arbitrary depth and non-linearities "with dropout applied before every weight" layer
- p. 4, §4, ¶3, the four lines after Eq. 6 (left column): the name and the procedure; "Monte Carlo estimate as MC dropout", "is equivalent to performing T stochastic forward passes"
- p. 4, §4, the two lines after the variance equation at the top of the right column: the predictive variance "equals the sample variance of T stochastic forward" passes plus the inverse model precision
- p. 7, Table 1 and §5.3, ¶3, lines 1-4 (left column): dropout "outperforms all other models both in terms of RMSE as" well as test log-likelihood on all datasets except Yacht (e.g. Boston Housing RMSE 4.32 VI, 3.01 PBP, 2.97 Dropout)

### the one result or idea that matters to us

The method needs two things: a network that was trained with dropout, and repeated forward passes at test time with the dropout masks resampled; the uncertainty is the variance across those passes. The trained model is not changed or retrained, which is why the method is attractive for an existing checkpoint, but it only applies if that checkpoint has dropout.

- p. 4, §4, last paragraph, lines 1-5 (right column): "Note that the dropout NN model itself is not changed", and the information "used with existing NN models trained with dropout"
- p. 3, §3, paragraph after Eq. 1, lines 1-5 (left column): what dropout is during training; "we sample binary variables for every input", and a unit is dropped when its variable is 0
- p. 4, §4, the two lines after the variance equation at the top of the right column: uncertainty is the "sample variance of T stochastic forward" passes (plus a constant term)
- p. 6, §5.1, last paragraph, lines 1-5 (left column): how many passes; T "was 1000 for drawing purposes", and T = 10 already gives a reasonable estimate (Fig. 3, p. 5)
- p. 7, last paragraph of the right column, lines 6-8, continuing on p. 8, left column, lines 1-3: the only place the paper treats a network without dropout; "repeating the same experiment with dropout probability 0" gives RMSE 3.07 and log-likelihood -2.59 on Boston Housing, against 2.97 and -2.46 with dropout (Table 1)

### how it relates to our design

MC dropout belongs to the family of methods that sample a stochastic depth model, and our report drops it as a baseline: main.tex says every dropout rate in the Depth Anything V2 encoder is zero and the decoder has none, so repeated passes are identical, and flip disagreement covers its role. The paper supplies the requirement that our backbone fails (a network trained with dropout, sampled at test time); the fact about the backbone is ours, not the paper's.

- p. 4, §4, last paragraph, lines 4-5 (right column): the method is for "existing NN models trained with dropout"
- p. 4, §4, ¶3, the lines after Eq. 6 (left column): the estimate is "T stochastic forward passes" averaged; with no dropout there is nothing stochastic to sample
- [progress.tex:48](../latex/sections/progress.tex:48): "MC dropout is dropped because every dropout rate in the Depth Anything V2 encoder is zero and the decoder has none, so repeated passes are identical; flip disagreement covers its role"
- [literature-review.tex:31](../latex/sections/literature-review.tex:31): most uncertainty work for depth changes the depth model, including one that "samples a stochastic one"; our backbone is frozen, so we are closest to post-hoc methods
- [literature-review.tex:35](../latex/sections/literature-review.tex:35): the \todo stub this sheet serves ("why it is a baseline in the proposal and why it is dropped")
- Absence claim, not quotable: the paper never says that a network without dropout gives identical passes or zero variance. It follows from the variance formula (p. 4, top of right column), but the sentence has to be ours. The paper's only dropout-probability-0 experiment (p. 7-8, above) retrains the network and reports accuracy, not the spread of the passes.

### notes for the writer

- Neither the paper nor main.tex says why MC dropout was a baseline in the proposal; main.tex only records that it was dropped ([progress.tex:48](../latex/sections/progress.tex:48)). The paper supports a generic reason (uncertainty from an existing dropout model without changing it, p. 4, §4, last paragraph), but the proposal's actual reason has to come from the proposal. Also, the stub at literature-review.tex:35 says the backbone "has no dropout", while progress.tex:48 says the encoder's dropout rates are zero and the decoder has none; I did not check the backbone code, so use whichever wording the code supports.
- Absence claim, not quotable: the paper has no depth, dense-prediction or driving experiment (pp. 1-9: CO2 and solar regression, MNIST with LeNet, UCI-style regression in Table 1, a 2D reinforcement-learning task). Do not cite it for how well MC dropout works on depth. Its cost statement, "constant running time identical to that of standard" dropout (p. 4, §4, last paragraph, lines 6-8, right column), assumes the T passes run concurrently; it is still T forward passes.
- `references.bib` agrees with the PDF: the p. 1 footer reads "Proceedings of the 33rd International Conference on Machine Learning, New York, NY, USA, 2016". The PDF is arXiv v6 (4 Oct 2016), which adds footnotes 3 and 4 on p. 7 correcting the standard errors in Table 1, so the Table 1 numbers here may differ from the proceedings version.

## post-review discussion with writer
