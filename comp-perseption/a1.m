%   A1.m  (posted)
%
%   author:  Michael Langer
%
%   COMP 546,  Winter 2017
%   Assignment 1 

close all
clear
blurWidth = 128;

% checkboard
N = 10;
s = 10;
I = makecheckerboard(N);
madeImage(I, 'Checkboard 10x10');
I(1:N*s,N*s/2+1:N*s,1) = 0.2*I(1:N*s,N*s/2+1:N*s,1);
I(1:N*s,N*s/2+1:N*s,2) = 0.2*I(1:N*s,N*s/2+1:N*s,2);
I(1:N*s,N*s/2+1:N*s,3) = 0.2*I(1:N*s,N*s/2+1:N*s,3);
madeImage(I, 'Checkboard 10x10 Shaded');
% local contrast for checkboard
for i=4:2:10
    g = make2DGaussian(blurWidth, i);
    cI = localContrast(I, g);
    madeImage(cI, sprintf('Checkboard 10x10 Shaded - Local Contrast sig=%d', i));
end
% % illusions
N = 256;
for i=4:4:20
    g = make2DGaussian(blurWidth, i);
    I =  makeImageSimultaneousContrast(N);
    cI = localContrast(I, g);
    madeImage(cI, sprintf('Illusion Local Contrast sig=%d', i));
    I =  imread('Whites_illusion.jpg');
    cI = localContrast(I, g);
    madeImage(cI, sprintf('Illusion Local Contrast sig=%d', i));
end
% plot for DOG
% stripes
dog = makeDOG(blurWidth,10,20);
samples= 50;
response = ones(1,samples+1);
for i=0:samples
    I = makeStipes(i+2);
    cI = dogResponse(I, dog, 1);
    % take the middle response because on the edge, it's wrong due to by the
    % edge of the image.
    response(i+1) = max(max(cI));
end
figure
plot([2:samples+2], response);
title('Response vs stripe size');
xlabel('stripe size');
ylabel('peak response');
I = makeStipes(33);
cI = dogResponse(I, dog, 0);
madeImage(I, sprintf('Peak response for stripes'));
madeImage(cI, sprintf('Peak response for stripes'));
% square 
dog = makeDOG(blurWidth,10,20);
samples= 50;
response = ones(1,samples+1);
for i=0:samples
    I = makeSquare(i+30);
    cI = dogResponse(I, dog, 1);
    response(i+1) = max(max(cI));
end
figure
plot([30:samples+30], response);
title('Response vs square size');
xlabel('square size');
ylabel('peak response');
I = makeSquare(55);
cI = dogResponse(I, dog, 0);
madeImage(I, sprintf('Peak response for square'));
madeImage(cI, sprintf('Peak response for square'));
% R-G opponency
N = 10;
I = randomCheckboardWithNoBlue(N);
rgopp = rgOpp(I, 0, 0, 0);
g = make2DGaussian(blurWidth, 5);
rgopp_local = rgOpp(I, g, g, 0);
madeImage(I, 'Normal checkboard');
madeImage(rgopp, 'R-G opponency');
madeImage(rgopp_local, 'R-G opponency with local normalization');
% using DOG
g1 = make2DGaussian(blurWidth, 2);
g2 = make2DGaussian(blurWidth, 4);
rgopp = rgOpp(I, g1, g2, 0);
madeImage(rgopp, 'R-G opponency with DOG');
% image of high red-gree contrast 
I =  imread('r-g.jpg');
rgopp = rgOpp(I, g1, g2, 0);
madeImage(rgopp, 'lady bugs');
rgopp = rgOpp(I, g1, g2, 1);
madeImage(rgopp, 'lady bugs');


function newI = normalizeToImage(I)
    max_I = max(max(I));
    min_I = min(min(I));
    newI = (I - min_I) ./ (max_I - min_I);
end

function newI = localContrast(I, g)
    %%

    %  Here we define the intensity, red-green, and blue-yellow axes
    %  These are unit vectors;

    R = squeeze( double( I(:,:,1) ) );    % 'squeeze' is useful for having compatible matrix sizes. 
    G = squeeze( double( I(:,:,2) ) );
    B = squeeze( double( I(:,:,3) ) );
    intensity = (R + G + B)/3;

    %%  Compute local contrast

    N = size(I);
    Nx = N(1);
    Ny = N(2);
    
    Rlocalmean  = filter2( g, R ); 
    Glocalmean  = filter2( g, G );
    Blocalmean  = filter2( g, B );
    intensityLocalMean = (Rlocalmean + Glocalmean + Blocalmean)/3;
    % The ./  operator divides pointwise.
    localcontrast  =    (intensity - intensityLocalMean) ./ intensityLocalMean;
    % normalize to output as picture
%     new_localcontrast = normalizeToImage(localcontrast);
    new_localcontrast = localcontrast;
    newI = ones(Nx, Ny, 3);
    newI(1:Nx, 1:Ny, 1) = new_localcontrast;
    newI(1:Nx, 1:Ny, 2) = new_localcontrast;
    newI(1:Nx, 1:Ny, 3) = new_localcontrast;
end

function d = madeImage(I, t)
    figure
    image(I);
    title(t);
    d = 1;
end

% make stipes for question 5
function I = makeStipes(s)
    N = max(10, ceil(100/s));
    I = ones(N*s, 100, 3);
    for i = 1:2*s:N*s
        I(i:i+s-1, 1:100, 1) = zeros(s,100);
        I(i:i+s-1, 1:100, 2) = zeros(s,100);
        I(i:i+s-1, 1:100, 3) = zeros(s,100);
    end
end

% make a square on a uniform background
function I = makeSquare(size)
    N = 100;
    I = ones(N, N, 3);
    start = floor((N-size)/2);
    ending = start+size-1;
    square = size;
    I(start:ending, start:ending, 1) = zeros(square,square);
    I(start:ending, start:ending, 2) = zeros(square,square);
    I(start:ending, start:ending, 3) = zeros(square,square);
end

% make DOG
function dog = makeDOG(w, sig1, sig2)
    dog1 = make2DGaussian(w, sig1);
    dog2 = make2DGaussian(w, sig2);
    dog = dog1 - dog2;
end

function newI = dogResponse(I, f, justResponse)
    R = squeeze( double( I(:,:,1) ) );
    G = squeeze( double( I(:,:,2) ) );
    B = squeeze( double( I(:,:,3) ) );
    intensity = (R + G + B)/3;

    response  = filter2( f, intensity );
    
    N = size(I);
    Nx = N(1);
    Ny = N(2);
    
    if justResponse == 1
        newI = response;
    % Normalize to output to image 
%   res = normalizeToImage(response);
    else
        res = response;
        newI = ones(Nx, Ny, 3);
        newI(1:Nx, 1:Ny, 1) = res;
        newI(1:Nx, 1:Ny, 2) = res;
        newI(1:Nx, 1:Ny, 3) = res;
    end
end

% R-G opponency
function I = randomCheckboardWithNoBlue(N)
    I = makecheckerboard(N);
    I(1:N*10, 1:N*10, 3) = zeros(N*10, N*10);
end

function newI = rgOpp(I, g1, g2, rec)
    R = squeeze( double( I(:,:,1) ) );
    G = squeeze( double( I(:,:,2) ) );
    rgopp = R - G;

    if g1 == 0
        localrgmean = (R + G)/2;
    else
        Rlocalmean  = filter2( g1, R ); 
        Glocalmean  = filter2( g2, G );
        localrgmean = (Rlocalmean + Glocalmean)/2;
    end
    localrgopp  = (rgopp - localrgmean) ./ localrgmean;
    
    N = size(I);
    Nx = N(1);
    Ny = N(2);
    
    if rec == 1
        % rectification
        new_rgopp = max(zeros(Nx, Ny), localrgopp);
    else
        new_rgopp = normalizeToImage(localrgopp);
    end
    
    N = size(I);
    Nx = N(1);
    Ny = N(2);
    newI = ones(Nx, Ny, 3);
    newI(1:Nx, 1:Ny, 1) = new_rgopp;
    newI(1:Nx, 1:Ny, 2) = new_rgopp;
    newI(1:Nx, 1:Ny, 3) = new_rgopp;
end
