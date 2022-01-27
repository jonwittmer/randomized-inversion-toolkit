X-ray Tomography is the mildly ill-posed inverse problem of reconstructing a 2D image from x-ray measurements taken from various angles. The governing model is the [radon transform](https://en.wikipedia.org/wiki/Radon_transform). We use the [scikit-image implementation](https://scikit-image.org/docs/dev/auto_examples/transform/plot_radon_transform.html) to generate synthetic measurements and generate the forward model. This problem differs from the _regularization toolbox_ examples in that the forward model is formed implicitly - this problem demonstrates a matrix-free method for solving inverse problems using the conjugate gradient method. 