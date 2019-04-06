classdef Reservoir < handle
    properties
        M; %input nodes
        N; %reservoir nodes
        P; %output nodes
        degree_func = @(x) sqrt(x);
        seed = 42;
        sigma = 1;
        bisa = 1;
        alpha = 1;
        beta = 1e-07;
        total_len;
        train_len;
        u;
        s;
        r;
        Win;
    end
    
    methods
        function self = Reservoir(input, output, N)
            self.M = size(input, 1);
            self.P = size(output, 1);
            self.N = N;
            self.total_len = size(input, 2);
            self.train_len = 0.7 * self.total_len;
            rng(self.seed);
            self.u = input;
            self.s = output;
            self.r = zeros(N, self.train_len);
            self.Win = -self.sigma + 2*self.sigma*rand(self.N, self.train_len);
           
            
            
        end
    end
end
