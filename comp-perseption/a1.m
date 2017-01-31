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
g = make2DGaussian(blurWidth, 10);
cI = localContrast(I, g);
madeImage(cI, 'Checkboard 10x10 Shaded - Local Contrast sig=10');
% % illusions
N = 256;
g = make2DGaussian(blurWidth, 10);
I =  makeImageSimultaneousContrast(N);
cI = localContrast(I, g);
madeImage(cI, 'Illusion Local Contrast sig=10');
I =  imread('Whites_illusion.jpg');
cI = localContrast(I, g);
madeImage(cI, 'Illusion Local Contrast sig=10');
% DOG
I = makeStipes();
dog = makeDOG(5,1,2);
cI = dogResponse(I, dog, 1);
madeImage(I, 'Stipes');
madeImage(cI, 'Stipes DOG sig=1,4 w=5');
% plot with square
dog = makeDOG(30,30,60);
samples= 50;
response = ones(1,samples+1);
for i=0:samples
    I = makeSquare(i);
    max_res = dogResponse(I, dog, 0);
    response(i+1) = max_res;
end
figure
plot([11:2:samples*2+11], response);
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
    new_localcontrast = normalizeToImage(localcontrast);
    newI = ones(Nx, Ny, 3);
    newI(1:Nx, 1:Ny, 1) = new_localcontrast;
    newI(1:Nx, 1:Ny, 2) = new_localcontrast;
    newI(1:Nx, 1:Ny, 3) = new_localcontrast;
end

function d = madeImage(I, t)
    figure
    image(I);
    title(t)
    d = 1;
end

% make stipes for question 5
function I = makeStipes()
    N = 10;
    s = 10;
    I = ones(N*s, N*s, 3);
    for i = 1:2*s:N*s
        I(i:i+s-1, 1:N*s, 1) = zeros(s,N*s);
        I(i:i+s-1, 1:N*s, 2) = zeros(s,N*s);
        I(i:i+s-1, 1:N*s, 3) = zeros(s,N*s);
    end
end

% make a square on a uniform background
function I = makeSquare(size)
    s = 11;
    I = ones(s*s, s*s, 3);
    start = s*(s/2-0.5)-size;
    ending = s*(s/2+0.5)-1+size;    
    square = s+2*size;
    I(start:ending, start:ending, 1) = 0.8*ones(square,square);
    I(start:ending, start:ending, 2) = 0.8*ones(square,square);
    I(start:ending, start:ending, 3) = 0.8*ones(square,square);
end

% make DOG
function dog = makeDOG(w, sig1, sig2)
    dog1 = make2DGaussian(w, sig1);
    dog2 = make2DGaussian(w, sig2);
    dog = dog1 - dog2;
end

function newI = dogResponse(I, f, normalize)
    R = squeeze( double( I(:,:,1) ) );
    G = squeeze( double( I(:,:,2) ) );
    B = squeeze( double( I(:,:,3) ) );
    intensity = (R + G + B)/3;

    response  = filter2( f, intensity );
    
    N = size(I);
    Nx = N(1);
    Ny = N(2);
    
    if normalize == 1
        % Normalize to output to image 
        res = normalizeToImage(response);

        newI = ones(Nx, Ny, 3);
        newI(1:Nx, 1:Ny, 1) = res;
        newI(1:Nx, 1:Ny, 2) = res;
        newI(1:Nx, 1:Ny, 3) = res;
    else
        newI = max(max(response));
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
