clear 
close all
N = 256;

%  number of families of disparity tuned Gabors 
numdisparities = 17;   
%  e.g.  -8 to 8,  or -16 to 16 if we step by 2,  -32 to 32 if we step by 4
step = 3;
disparities = ((1:numdisparities) - 9)*step;

M = 32;    % window width on which Gabor is defined
k = 2;    % frequency
%  wavelength of underlying sinusoid is M/k pixels per cycle.

if (1) % put 0 to get horizontally oriented gabor
    [cosGabor, sinGabor] = make2DGabor(M,k,0);
else
    [cosGabor, sinGabor] = make2DGabor(M,0,k);
end
%  Make a random dot stereogram with a central square protruding from a 
%  plane.  The plane has disparity = 0.  The central square has some positive
%  disparity.

%  We will choose our intensities to have mean 0.  The reason is that a cos Gabor 
%  doesn't sum to zero -- one can show that it always has a positive response 
%  to a constant intensity image. 
%  If we were to make the intensities random in 0 to 1,  then we would bias
%  the cosGabor responses to be positive, which is not what we want.

disparitySquare = 4;

Ileft = rand(N,N) - .5;
Iright = Ileft;
Iright(N/4:3*N/4, N/4:3*N/4) = Ileft(N/4:3*N/4, N/4  + disparitySquare: ...
        3*N/4 + disparitySquare);
Iright(N/4:3*N/4, 3*N/4+1:3*N/4+disparitySquare) = rand(N/2+1, disparitySquare) - 0.5;

% Show the image as an anaglyph

figure
I = zeros(N,N,3);
I(:,:,1) = Ileft;       %  put left eye image into the R channel
I(:,:,2) = Iright;      %  put right eye image into the G and B channels
I(:,:,3) = Iright;

image(I);
title(['disparity is ' num2str(disparitySquare)  ' pixels'])
axis square
axis off

%  Here comes the fun stuff... 
%  To define disparity tuned Gabor cell, we want to shift the left  
%  Gabor cells relative to the right one. 
%  If the shift is 0, then the cell is tuned to 0 
%  disparity as was the case looked at in the lecture.  If we want 
%  a different disparity then we need to shift by some other amount
%  which I am calling d_Gabor.
%
%  The binocular complex cell's response at (x,y) is the length of the vector
%  that is defined by the sum of the responses of the (cos, sin) Gabors   
%  of the left and right eye Gabor cells.   To compute 
%  the binocular complex cell response, take the cross correlation of the sin 
%  and cos Gabors with the left image and right images -- four cross correlations needed.    
%  To define a binocular complex cell that has a preferred 
%  disparity,  we need to combine the responses of one eye with shifted responses of the other eye.

 
figure
responses = zeros(N,N,numdisparities);    %  compute responses for all different disparities

% Ileft_filtered_cos  = filter2( cosGabor, Ileft, 'same');
% Ileft_filtered_sin  = filter2( sinGabor, Ileft, 'same');
Iright_filtered_cos = filter2( cosGabor, Iright, 'same');
Iright_filtered_sin = filter2( sinGabor, Iright, 'same');
% makeImage(cosGabor, 'cosine gabor');
% makeImage(sinGabor, 'sine gabor');

%%%---------   ADD YOUR CODE BELOW HERE    
for i=1:numel(disparities)
    d = disparities(i);
    d_cosGabor = circshift(cosGabor,d,2);
    d_sinGabor = circshift(sinGabor,d,2);
    if d == 16
        makeImage(d_cosGabor, 'cosine gabor, shiftedby 16 px');
        makeImage(d_sinGabor, 'sine gabor, shifted by 16 px');
    end
    Ileft_filtered_cos  = filter2( d_cosGabor, Ileft, 'same');
    Ileft_filtered_sin  = filter2( d_sinGabor, Ileft, 'same');
    if (1) % put 0 here to get difference instead of sum response
        responses(1:N, 1:N, i) = (Ileft_filtered_cos + Iright_filtered_cos).^2 + (Ileft_filtered_sin + Iright_filtered_sin).^2;
    else
        responses(1:N, 1:N, i) = (Ileft_filtered_cos - Iright_filtered_cos).^2 + (Ileft_filtered_sin - Iright_filtered_sin).^2;
    end
end


%%%----------   ADD YOUR CODE ABOVE HERE 

% Here you can "normalize" the responses.   I am 'commenting' out this code for now,
% but I suggest you consider it when trying to understand how responses of various
% Gabor families depend on disparity and on the orientation of the Gabor.

if (1)
    meanResponses = mean(responses,3);
    for j = 1:numdisparities
        responses(:,:,j) = (responses(:,:,j) - meanResponses) ./ meanResponses; 
    end
end

%  Here is some plotting code that you can use for Q2(a).
%  'responses' is the responses of a family of binocular disparity tuned complex
%  Gabor cells.  Each family has particular peak disparity to which it is tuned.    
%  'responses' is an N x N x 17 matrix

for j = 1:numdisparities
    if mod(j, 2) == 1  % image plot only for odd numbered indices of disparities
        subplot(3,3, (j+1)/2);   %  3x3 plot
        subimage( remapImageUint8( squeeze(responses(:,:,j))) );
        colormap gray
        title( ['tuned to d = ' num2str(disparities(j))  ] );
        axis off
    end
end


sumResponseCenter    = zeros(numdisparities,1);
meanResponseSurround = zeros(numdisparities,1);

for j = 1:numdisparities
  center =  responses(N/4:3*N/4, N/4:3*N/4,j);
  sumResponseCenter(j) = sum(center(:));
end
meanResponseCenter = sumResponseCenter / power(N/2 + 1,2); 

figure
plot(disparities, meanResponseCenter,'-*b'); 
hold on

for j = 1:numdisparities
  meanResponseSurround(j) = (sum(sum(responses(:, :,j))) - sumResponseCenter(j)) /  (N*N - power(N/2+1,2));
end
plot(disparities, meanResponseSurround,'*-r');
legend(['mean response in center square (disparity = ' num2str(disparitySquare) ') '], 'mean response in background (disparity = 0)');
xlabel('disparity tuning of Gabor family','FontSize',12);
ylabel('mean response of Gabor family ','FontSize',12);
title(['wavelength = ' num2str(M/k) ' ,  sigma = ' num2str(M/k/2) ' (pixels)'],'FontSize',12);


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
