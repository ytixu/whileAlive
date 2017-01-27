close all
clear
blurWidth = 128;

if (0)
   N = 256;
   I =  makeImageSimultaneousContrast(N);
else
    I =  imread('Whites_illusion.jpg');
end

%I = makecheckerboard(10);

figure
image(I);
title('some original RGB image')

%%

%  Here we define the intensity, red-green, and blue-yellow axes
%  These are unit vectors;

R = squeeze( double( I(:,:,1) ) );    % 'squeeze' is useful for having compatible matrix sizes. 
G = squeeze( double( I(:,:,2) ) );
B = squeeze( double( I(:,:,3) ) );
intensity = (R + G + B)/3;

%%  Compute local contrast

N = size(I)
Nx = N(1)
Ny = N(2)
sig = 30;
g = make2DGaussian(blurWidth, sig);
Rlocalmean  = filter2( g, R ); 
Glocalmean  = filter2( g, G );
Blocalmean  = filter2( g, B );
intensityLocalMean = (Rlocalmean + Glocalmean + Blocalmean)/3;
% The ./  operator divides pointwise.
localcontrast  =    (intensity - intensityLocalMean) ./ intensityLocalMean;
localcontrast = localcontrast ./ max(localcontrast);
I = ones(Nx, Ny, 3);
I(1:Nx, 1:Ny, 1) = localcontrast;
I(1:Nx, 1:Ny, 2) = localcontrast;
I(1:Nx, 1:Ny, 3) = localcontrast;

figure
image(I);
title('Local contrast')
