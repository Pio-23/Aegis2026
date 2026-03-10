% Visualizes a text file of points. No PCD header required.

points = readmatrix("testcloud.txt");

% 2D ring visualizer
x = points(:, 1);
y = points(:, 2);
z = zeros(length(x), 1);
intensity = points(:, 3);
point_size = 1;

% 3D cloud visualizer
% x = points(:, 1);
% y = points(:, 2);
% z = points(:, 3);
% intensity = points(:, 4);
% point_size = 1;

scatter3(x, y, z, point_size, intensity)