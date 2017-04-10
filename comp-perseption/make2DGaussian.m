function gaussian2D = make2DGaussian(sig)
%
%  returns a NxN matrix which is a Gaussian with standard deviation sig 
%

gaussian1D = makeGaussian(sig);
gaussian2D = gaussian1D * gaussian1D';