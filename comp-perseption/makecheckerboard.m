function I = makecheckerboard(N)
    s = 10;
    I = ones(N*s, N*s, 3);
    for i = 1:s:N*s
        for j = 1:s:N*s
            I(i:i+s-1, j:j+s-1, 1) = rand(1)*ones(s,s);
            I(i:i+s-1, j:j+s-1, 2) = rand(1)*ones(s,s);
            I(i:i+s-1, j:j+s-1, 3) = rand(1)*ones(s,s);
        end
    end
