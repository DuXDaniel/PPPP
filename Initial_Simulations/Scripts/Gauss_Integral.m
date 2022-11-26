function Z = Gauss_Integral(func, xmin, xmax, N)
X = gpuArray.linspace(xmin, xmax, N);
xspacing = (xmax - xmin)/N;
F = arrayfun(func, X);
Z = trapz(F) * xspacing;
end