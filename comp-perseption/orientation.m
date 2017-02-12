clear;  close all


if (1)
    readImage  ; 
    sizeI = size(I);
    NX = sizeI(2);
    NY = sizeI(1);
else
%  Synthesize an image which is an annulus (ring).
%  The orientation detector will annulus from red (at horizontal) 
%  around the hue circle (ROYGBV), back to red (at 180 degrees
%  away, i.e. horizontal again.

    
    N = 512;
    NX = N;
    NY = N;
    I = zeros(N,N);
    minRadius = 110;
    maxRadius = 115;
    Iannulus = sqrt(power((-N/2:N/2-1)' * ones(1,N),2) + ...
             power(ones(N,1) * (-N/2:N/2-1),2));
    Iannulus = double( (Iannulus >= minRadius) & (Iannulus <= maxRadius) );
    I = Iannulus;
    I(N/2 - maxRadius: N/2 + maxRadius, N/2) = 1;  % thinner than the circle so response won't
    I(N/2, N/2 - maxRadius: N/2 + maxRadius) = 1;  % be as great 
end

%  Show the image (or the green channel only, if the image is color)

figure(1); imagesc(I); colormap(gray(256)); % colorbar;

%  Filter the image with N_THETA orientations, 0, 180/N_THETA, 
%  2*(180/N_THETA),  ... (N_THETA-1)*180/N_THETA  degrees.
%

N_THETA = 12;    

M = 32;    % window width on which Gabor is defined
k = 2;    % frequency
%  wavelength of underlying sinusoid is M/k pixels per cycle.

thetaRange = pi/N_THETA * (0:N_THETA-1);

% initialize some matrices to junk values

peakTheta   = zeros(NY,NX);
maxResponse = zeros(NY,NX);
minResponse = 10000*ones(NY,NX);
sumResponse = zeros(NY,NX);

%  Hint:  You can use the following trick to find the max
%         response and other required quantities.
%
%     for each theta
%        filter the image with a Gabor tuned to that theta
%  Matlab code:   mask = (filterResponse > maxResponse);
%  Matlab code:   maxResponse = mask .* filterResponse + ~mask .* maxResponse
%     end

%---------------  ADD YOUR CODE BELOW HERE ------------------
for i = 1:numel(thetaRange)
    theta = thetaRange(i);
    [cosGabor, sinGabor] = make2DGabor(M,sin(theta)*k,cos(theta)*k);
%    makeImage(cosGabor, 'cosgabor');
%    makeImage(sinGabor, 'singabor');
    cosResponse = filter2( cosGabor, I, 'same');
    sinResponse = filter2( sinGabor, I, 'same');
%     makeImage(cosResponse, 'cosgabor response');
%     makeImage(sinResponse, 'sinegabor response');
    % get measures
    filterResponse = (cosResponse.^2 + sinResponse.^2).^(1/2);

    mask = (filterResponse > maxResponse);
    maxResponse = mask .* filterResponse + ~mask .* maxResponse;    
    peakTheta = mask * theta + ~mask .* peakTheta; 
    
    mask = (filterResponse < minResponse);
    minResponse = mask .* filterResponse + ~mask .* minResponse;    
    
end
makeImage(maxResponse, 'max response');
makeImage(minResponse, 'min response');
makeImage(peakTheta, 'peak theta');

%---------------  ADD YOUR CODE ABOVE HERE ------------------

figure
%
hsvImage = zeros(NY,NX,3);
%  hue is orientation.   Required to be in 0 to 1. 
hsvImage(:,:,1) = peakTheta/180;  
%  saturation indicates whether there is a big difference between 
%  max and min
hsvImage(:,:,2) = (maxResponse-minResponse)./(maxResponse + minResponse);  % 

%  value indicates whether the response is large compared to the max
%  response over the image.   Just normalizes to 0 to 1 as required by hsv.
hsvImage(:,:,3) = maxResponse/max(maxResponse(:));

image(hsv2rgb(hsvImage));
%axis square

function newI = normalizeToImage(I)
    max_I = max(max(I));
    min_I = min(min(I));
    newI = (I - min_I) ./ (max_I - min_I);
end

function d = makeImage(I, t)
    N = size(I);
    Nx = N(1);
    Ny = N(2);
    I = normalizeToImage(I);
    newI = ones(Nx, Ny, 3);
    newI(1:Nx, 1:Ny, 1) = I;
    newI(1:Nx, 1:Ny, 2) = I;
    newI(1:Nx, 1:Ny, 3) = I;
    figure
    image(newI);
    title(t);
    d = 1;
end
