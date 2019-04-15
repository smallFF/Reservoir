classdef Reservoir < handle
    properties
        M;                   % input nodes
        N;                   % reservoir nodes
        P;                   % output nodes
        t;                   % time
        dataset_in;          % input data set
        dataset_out;         % output data set
        total_len;           % total data length
        init_len;            % record states after init_len
        train_len;           % training data length
        
        degree_func = @(x) sqrt(x);
        seed = 42;           % random seed
        sigma = 1;
        bisa = 1;
        alpha = 1;
        beta = 1e-07;
        
        u;                  % training input data set   
        s;                  % training output data set
        S;                  % predict dataset_out 
        r;                  % reservoir states
        R;                  % composition of u and r
        Win;                % input weights
        Wout;               % output weights
        W;                  % reservoir weights

        RMS;                % Root Mean Square errors
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
            self.init_len = 0;
            self.train_len = floor(0.5 * self.total_len);
            rng(self.seed);
            
            self.u = input(:, 1: self.train_len);
            self.s = output(:, self.init_len + 1: self.train_len);
            self.S = zeros(self.P, self.total_len);
            self.r = zeros(N, self.train_len - self.init_len);
            self.R = zeros(1 + self.M + self.N, self.train_len - self.init_len);
            self.Win = -self.sigma + 2*self.sigma*rand(self.N, 1 + self.M);
            self.Wout = [];
            
            W = rand(N,N) - 0.5;
            opt.disp = 0;
            rhoW = abs(eigs(W, 1, 'LM', opt));
            self.W = W .* (1.0/rhoW);
            
            self.RMS = zeros(self.P, 1);
        end
            
        function train(self)
%             uu = self.dataset_in(:, 1);
%             rr = zeros(self.N, 1);
            
            for i = 1:self.train_len - 1
               uu = self.dataset_in(:, i + 1);
               self.r(:, i + 1) = (1 - self.alpha) * self.r(:, i) + ...
                self.alpha * tanh( self.W * self.r(:,i) + self.Win *[1;uu]);
               rr = self.r(:, i + 1);
               
               if i >= self.init_len
                self.R(:, i + 1 - self.init_len) = [1; uu; rr];
               end
               
            end

            self.Wout = self.s * self.R' ...
                / (self.R * self.R' + self.beta * eye(1 + self.M + self.N));
        end
        
        function predict(self)
            RR = zeros(1 + self.M + self.N, self.total_len);
            rr = zeros(self.N, 1);
            for i = 1:self.total_len - 1
               uu = self.dataset_in(:, i + 1);
               rr = (1 - self.alpha) * rr + ...
                self.alpha * tanh( self.W * rr + self.Win *[1;uu]);
               RR(:, i + 1) = [1; uu; rr];
            end
            
            self.S = self.Wout * RR;
            
            self.RMS = sqrt(sum((self.dataset_out - self.S).^2 ,2) ./ self.total_len);
            
        end
        
        function draw(self)
           figure();
           for i = 1:self.P
              subplot(self.P, 1, i);
              plot(self.t, self.dataset_out(i,:), 'b-', ...
                  self.t, self.S(i,:),'r--')
              legend('true solution', 'predict solution');
              xlabel(['RMS error = ',num2str(self.RMS(i))]);
           end
        end
    end
end
