%  Assignment 4  COMP 546  Winter 2017
%
%  Create a simple random dot stereogram with left and right images
%  Ileft and Iright.  Filter the stereo pair with a family of 
%  binocular complex Gabors.  

clear 
close all

% question 1
if (0)
    O = [0, 45, 90, 135]; 
    M = [16, 32, 64]; 
    k = 4;
    beta = 1;

    thetaRange = pi/180 * O;

    for i = 1:numel(M)
        m = M(i);
        for j = 1:numel(thetaRange)
            t = thetaRange(j);
            sinGabor = make2DGabor(m,sin(t)*k,cos(t)*k, beta);
            makeImage(sinGabor, 'response', m);
        end
    end
end

% question 2

f1 = fopen('I.raw','r');
I = fread(f1,'float');
fclose(f1);
II = I;
m = 240;
n = size(I);

for i=1:n
   II(i) = mean(abs(I(max(i-m,1):min(i+m,n))));
end
figure
plot(II)

soundsc(cat(1, I(16000:19000), I(35000:38000), I(44000:47000), I(67000:70000))); 



function newI = normalizeToImage(I)
    max_I = max(max(I));
    min_I = min(min(I));
    newI = (I - min_I) ./ (max_I - min_I);
end

function d = makeImage(I, t, m)
    N = size(I);
    Nx = N(1);
    Ny = N(2);
    I = normalizeToImage(I);
    newI = ones(Nx, Ny, 3);
    newI(1:Nx, 1:Ny, 1) = I;
    newI(1:Nx, 1:Ny, 2) = I;
    newI(1:Nx, 1:Ny, 3) = I;
    figure
    img = image(newI);
    title(t);
    %imagesc(-m/2:m/2-1, -m/2:m/2-1, img );
    d = 1;
end