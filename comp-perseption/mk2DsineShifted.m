function I = mk2DsineShifted(M,N,KX,KY)
%  I = mk2DsineShifted(N,KX,KY);
%
%  This program displays a 2D sine wave sin( 2pi/N (KX (x - (N/2+1)) + KY (y - (N/2+1)))).
%  where x,y are in 0,1,..N-1.
%
%  Here the x are the column indices and y are the row indices, which is what we want.
%  (Often matlab notation has it the other way around, more consistent with Matrix A_ij notation 

%display('arguments should be (N,KX,KY)');
if nargin < 3
  return;
end

x = -N/2: N/2-1;
y = -N/2: N/2-1;
[X,Y] = meshgrid(x,y);
I = sin(  2*pi/M * (KX * X + KY * Y) );

if (0)
    imagesc(I)
    colormap('gray')
    axis('square')
    xlabel('Y (second coordinate i.e. row)')
    ylabel('X (first  coordinate i.e. column)')
end

