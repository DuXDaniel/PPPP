function [] = splitIntegral(func, limits, waypoints)

sum = 0;
bound_arr = [limits(1), waypoints, limits(2)];

for i = 1:length(bound_arr)-1
    sum = sum + integral(func, bound_arr(i), bound_arr(i+1));
end

end