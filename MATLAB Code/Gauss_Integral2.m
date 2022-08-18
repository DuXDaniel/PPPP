function Z = Gauss_Integral2(func, xmin, xmax, ymin, ymax, N)
xvals = gpuArray.linspace(xmin, xmax, N);
yvals = gpuArray.linspace(ymin, ymax, N);
[X, Y] = meshgrid(xvals, yvals);
xspacing = (xmax - xmin)/N;
yspacing = (ymax - ymin)/N;
F = arrayfun(func, X, Y);
Z1 = trapz(F) * yspacing;
Z = trapz(Z1) * xspacing;
end