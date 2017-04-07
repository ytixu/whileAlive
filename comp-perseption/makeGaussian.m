function gaussian = makeGaussian(sig)
%
%  returns a 1xN vector  Gaussian with standard deviation sig.

%  If sig of Gaussian is much less than a pixel width, then blurring
%  with such a Gaussian does nothing.  So don't bother.

if (sig < .3)
    gaussian = [1];
else
  N = round(9*max(sig,1));
  gaussian = zeros(N,1);
  gaussian(1:N) = exp( - power( (1:N) - N/2 - .5,2) / (2 * sig*sig));
  gaussian = gaussian / sum(gaussian);
end

