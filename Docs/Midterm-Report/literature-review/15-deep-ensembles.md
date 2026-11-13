# literature review evidence: [Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles](../../Papers/1612.01474.pdf)

Lakshminarayanan, Pritzel and Blundell, NIPS (NeurIPS) 2017 · bib key `lakshminarayanan2017ensembles` · report §2.2

## ai-generated summary review

### what the paper does

It proposes a non-Bayesian recipe for predictive uncertainty: train each network with a proper scoring rule (for regression, a network that outputs a mean and a variance trained with Gaussian negative log-likelihood), optionally add adversarial training, and train an ensemble of M such networks whose predictions are averaged. On regression and classification benchmarks it reports uncertainty as good as or better than approximate Bayesian networks and MC-dropout.

- p. 1, Abstract, lines 7-9: "We propose an alternative to Bayesian NNs", simple to implement, readily parallelizable, little hyperparameter tuning
- p. 3, §2.1, ¶2, lines 1-2: the three-step recipe, ending "and (3) train an ensemble"
- p. 3, §2.2.1, ¶1, lines 3-6 and Eq. 1: the network "outputs two values in the final layer", mean and variance, trained with negative log-likelihood
- p. 5, paragraph after Algorithm 1, lines 1-3: the ensemble is "a uniformly-weighted mixture model"; for classification this is averaging the predicted probabilities
- p. 2, paragraph "Summary of contributions", lines 8-9: "significantly outperforms (or matches) MC-dropout" and "our simple yet strong baseline"

### the one result or idea that matters to us

The method needs M separately trained networks: each one starts from its own random initialisation and is trained on the full training set (recommended default M = 5), and the uncertainty comes from combining them. The only costs the paper states are that this is M trainings (independent, so parallelizable) and that the ensemble has M times the parameters of one network.

- p. 5, Algorithm 1, steps 1-3: "Recommended default values are M = 5"; step 2 "Initialize θ1, θ2, . . . , θM randomly"; step 3 comment "train networks independently in parallel"
- p. 4, §2.4, ¶2, lines 1-4: "We used the entire training dataset to train each network"; "random initialization of the NN parameters, along with random shuffling of the data points" was sufficient; bagging made results worse
- p. 9, §4, ¶2, lines 4-5: "The ensemble has M times more parameters than a single network" (followed by distillation as the remedy for memory-constrained applications)
- p. 9, §4, ¶1, lines 7-8: "Ensembles, even for M = 5, significantly improve uncertainty quality in all the cases"
- p. 7, §3.4, paragraph starting "Finally, we evaluate on the ImageNet", line 2: "Due to computational constraints, we only evaluate the effect of ensembles" (the one place the authors report being limited by compute)

### how it relates to our design

Our backbone is one stock Depth Anything V2 checkpoint that we keep frozen, so we have a single set of depth-model weights and cannot produce the M independently initialised and trained copies this method averages over. Our report places the paper among the approaches that change the depth model ("trains several models"), which our setting rules out.

- p. 5, Algorithm 1, steps 2-3: M randomly initialised networks, each trained (what the method requires)
- p. 3, §2.1, ¶2, lines 2-3: "Let M denote the number of NNs in the ensemble"
- [main.tex:97](../latex/main.tex:97): §2.2 opening; uncertainty work for depth "adds a variance output, trains several models, or samples a stochastic one", and "Our setting forbids that, since the backbone is frozen"
- [main.tex:151](../latex/main.tex:151): backbone paragraph; "The stock relative Depth Anything V2 ViT-B checkpoint", frozen, run once per image
- [main.tex:103](../latex/main.tex:103): the \todo stub for this paper: "the strong but expensive baseline; not applicable to a single frozen checkpoint"

### notes for the writer

- "Strong baseline" is the paper's own phrase (p. 2, "simple yet strong baseline"; p. 9, §4, ¶1, line 1, "a very strong baseline"). "Expensive" is not: the paper uses "computationally expensive" only for Bayesian NNs (p. 1, Abstract, lines 5-6) and presents its own method as "simple and scalable". The quotable costs are M trained networks and "M times more parameters" (p. 9). Absence claim, not quotable: the paper never states the inference cost (M forward passes per input), so if the report says "expensive", attribute it to us or word it as "requires training and storing M networks".
- "Not applicable to a single frozen checkpoint" is a statement about our setting, not something the paper says. Absence claim, not quotable: pretrained, frozen or downloaded models are not discussed anywhere in pp. 1-9; every ensemble member in the paper is trained from scratch. The requirement also only rules out ensembling the backbone; main.tex gives no statement on ensembling the trainable head, so avoid wording that implies ensembles are impossible anywhere in our pipeline.
- The experiments are small-scale regression (toy 1-D and tabular datasets, 1-hidden-layer networks; p. 6) and image classification (MNIST, SVHN, ImageNet; pp. 6-8). Absence claim, not quotable: there is no depth estimation or any dense per-pixel task, so do not cite this paper for ensemble results on depth.

## post-review discussion with writer
