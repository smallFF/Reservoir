classdef Reservoir < handle
    properties
        M; %input nodes
        N; %reservoir nodes
        P; %output nodes
        dataset_in;
        dataset_out;
        y_hat;
        degree_func = @(x) sqrt(x);
        seed = 42;
        sigma = 1;
        bisa = 1;
        alpha = 1;
        beta = 1e-07;
        total_len;
        train_len;
        t;
        u;
        s;
        r;
        Win;
        Wout;
        W;
        S;
    end
    
    methods
        function self = Reservoir(input, output, t, N)
            self.M = size(input, 1);
            self.P = size(output, 1);
            self.N = N;
            self.t = t;
            
            self.dataset_in = input;
            self.dataset_out = output;
            
            self.total_len = size(input, 2);
            self.train_len = floor(0.7 * self.total_len);
            rng(self.seed);
            self.u = input(:, 1: self.train_len);
            self.s = output(:, 1: self.train_len);
            self.r = zeros(N, self.train_len);
            self.Win = -self.sigma + 2*self.sigma*rand(self.N, 1 + self.M);
            self.S = zeros(1 + self.M + self.N, self.train_len);
            W = rand(N,N) - 0.5;
            opt.disp = 0;
            rhoW = abs(eigs(W, 1, 'LM', opt));
            self.W = W .* (1.25/rhoW);
            self.Wout = [];
        end
            
        function train(self)
%             uu = self.dataset_in(:, 1);
%             rr = zeros(self.N, 1);
            
            for i = 1:self.train_len - 1
               uu = self.dataset_in(:, i + 1);
               self.r(:, i + 1) = (1 - self.alpha) * self.r(:, i) + ...
                self.alpha * tanh( self.W * self.r(:,i) + self.Win *[1;uu]);
               rr = self.r(:, i + 1); 
               self.S(:, i + 1) = [1; uu; rr];
            end
            disp('S : size()')
            disp(size(self.S))
            disp('S : size()')
            disp(size(self.S'))
            self.Wout = self.s * self.S' ...
                / (self.S * self.S' + self.beta * eye(1 + self.M + self.N));
        end
        
        function predict(self)
            S = zeros(1 + self.M + self.N, self.total_len);
            rr = zeros(self.N, 1);
            for i = 1:self.total_len - 1
               uu = self.dataset_in(:, i + 1);
               rr = (1 - self.alpha) * rr + ...
                self.alpha * tanh( self.W * rr + self.Win *[1;uu]);
               S(:, i + 1) = [1; uu; rr];
            end
            
            self.y_hat = self.Wout * S;
        end
        
        function draw(self)
           for i = 1:self.P
              figure(i)
              plot(self.t, self.dataset_out(i,:), 'b-', ...
                  self.t, self.y_hat(i,:),'r--')
              legend('true solution', 'predict solution');
           end
        end
    end
end
