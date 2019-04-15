classdef Reservoir < handle
    properties
        model_name;          % model name
        M;                   % input nodes
        N;                   % reservoir nodes
        P;                   % output nodes
        t;                   % time
        dataset_in;          % input data set
        dataset_out;         % output data set
        total_len;           % total data length
        init_len;            % record states after init_len
        train_len;           % training data length
        
%         degree_func = @(x) sqrt(x);
        seed = 42;           % random seed
        sigma = 1;
%         bias = 1;
        alpha = 1.0;
        beta = 1e-08;
        
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
        function self = Reservoir(Model)
            self.model_name = Model.model.name;
            input = Model.model.states(:, 1)';
            output = Model.model.states(:, 2:3)';
            
            self.M = size(input, 1);
            self.P = size(output, 1);
            self.N = Model.model.N;
            self.t = Model.model.t;
            
            self.dataset_in = input;
            self.dataset_out = output;
            
            self.total_len = size(input, 2);
            self.init_len = 0;
            self.train_len = floor(0.5 * self.total_len);
            rng(self.seed);
            
            self.u = input(:, self.init_len + 1: self.train_len);
            self.s = output(:, self.init_len + 1: self.train_len);
            self.S = zeros(self.P, self.total_len);
            self.r = zeros(self.N, self.train_len - self.init_len);
            self.R = zeros(1 + self.M + self.N, self.train_len - self.init_len);
            self.Win = -self.sigma + 2*self.sigma*rand(self.N, 1 + self.M);
            self.Wout = [];
            
            W = 2*rand(self.N,self.N) - 1;
            opt.disp = 0;
            rhoW = abs(eigs(W, 1, 'LM', opt));
            self.W = W .* (1.0/rhoW);
            
            self.RMS = zeros(self.P, 1);
        end
            
        function train(self)
            
            rr = zeros(self.N, 1);
            
            for i = 1:self.train_len
               uu = self.dataset_in(:, i);
%                self.r(:, i + 1) = (1 - self.alpha) * self.r(:, i) + ...
%                 self.alpha * tanh( self.W * self.r(:,i) + self.Win *[1;uu]);
%                rr = self.r(:, i + 1);

               rr = (1 - self.alpha) * rr + ...
                self.alpha * tanh( self.W * rr + self.Win *[1;uu]);
               
               if i > self.init_len
                   self.r(:, i - self.init_len) = rr;
                   self.R(:, i - self.init_len) = [1; uu; rr];
               end
               
            end

            self.Wout = self.s * self.R' ...
                / (self.R * self.R' + self.beta * eye(1 + self.M + self.N));
        end
        
        function predict(self)
            RR = zeros(1 + self.M + self.N, self.total_len);
            rr = zeros(self.N, 1);
            for i = 1:self.total_len
               uu = self.dataset_in(:, i);
               rr = (1 - self.alpha) * rr + ...
                self.alpha * tanh( self.W * rr + self.Win *[1;uu]);
               RR(:, i) = [1; uu; rr];
            end
            
            self.S = self.Wout * RR;
            
            self.RMS = sqrt(sum((self.dataset_out - self.S).^2 ,2) ./ self.total_len);
            
        end
        
        function draw(self)
           figure();
           suptitle(self.model_name);
           for i = 1:self.P
              subplot(self.P, 1, i);
              plot(self.t, self.dataset_out(i,:), 'b-',...
                  self.t, self.S(i,:),'r--', 'LineWidth', 1)
              legend('true solution', 'predict solution');
              xlabel(['RMS error = ',num2str(self.RMS(i))]);
           end
%            saveas(gcf,['Matlab--',self.model_name,' N = ',num2str(self.N),'.png']);
           print(gcf,'-dpng','-r600',['Matlab--',self.model_name,' N = ',num2str(self.N),'.png'])
        end
        
        function disp(self)
           model.model_name = self.model_name;
           model.M = self.M;
           model.N = self.N;
           model.P = self.P;
           model.t = self.t;
           model.dataset_in = self.dataset_in;
           model.dataset_out = self.dataset_out;
           model.sigma = self.sigma;
%            model.bias = self.bias;
           model.alpha = self.alpha;
           model.beta = self.beta;
           model.Win = self.Win;
           model.W = self.W;
           model.Wout = self.Wout;
           model.RMS = self.RMS;
           
           disp(model);
        end
        function run(self)
           self.train()
           self.predict()
           self.draw()
        end
    end
end
