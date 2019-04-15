close all;
lorenz = @(t, X) [-10*(X(1) - X(2));
                  28*X(1) - X(2) - X(1)*X(3);
                  X(1)*X(2) - 8/3 * X(3)];
[time, y] = ode45(lorenz, 0:0.01:50, [1, 1, 1]);
% plot(time,y(:,1));
y = (y-mean(y,1)) ./ mean((y - mean(y,1)).^2,1).^(1/2);
input = y(:,1)';
output = y(:,2:3)';
N = 400;
t = time;
r = Reservoir(input, output, t, N);
r.train()
r.predict()
r.draw()
