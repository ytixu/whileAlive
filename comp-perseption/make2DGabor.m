function sinGabor = make2DGabor(M,k0,k1,beta)

N = 128;
k      =  sqrt(k0*k0 + k1*k1);
%sigma =  N/k/2
sigma  =  M* 2*pi/N *k*(2^beta-1)/sqrt(2*log(2))/(2^beta+1);
sin2D = mk2DsineShifted(M,N,k0,k1) ;

shiftx = ((1:N) - (N/2 + 1));
Gaussian = 1/(sqrt(2*pi)*sigma) * ...
    exp(- shiftx.*shiftx/ (2 * sigma*sigma) );
Gaussian2D = Gaussian' * Gaussian;

sinGabor = Gaussian2D .* sin2D;

