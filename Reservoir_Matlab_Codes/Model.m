classdef Model < handle
    properties
      % model structure
        model = struct('name', '',...
              'init_value', '',...
              't', '',...
              'args', '',...
              'N', 400,...
              'rho', 1.0,...
              'sigma', 1.0,...
              'bias', 1.0,...
              'alpha', 1.0,...
              'beta', 1e-8,...
              'states','');
        
        % % states data set 
        % states = [];
    end
    
    methods
        function self = Model(model_name)
%            model_list = 'lorenz-rossler';
%            if ~contains(model_list, model_name)
%                error("Not only 'lorenz' and 'rossler' model are supported!")
%            end
           
            if strcmp(model_name, 'lorenz')
                self.model.name = model_name;
                self.model.init_value = [0.1, 0.1, 0.1];
                self.model.t = 0 : 0.05 : 50;
                self.model.args = [10, 28, 8/3];
                self.model.N = 400;
                self.model.rho = 1.0;
                self.model.sigma = 1.0;
                self.model.bias = 1.0;
                self.model.alpha = 0.94;
                self.model.beta = 1e-8;
                [~, y] = ode45(@(t, X) self.lorenz(t, X, self.model.args), ...
                                       self.model.t, self.model.init_value);

            elseif strcmp(model_name, 'rossler')
                disp(['model_name =',model_name])
                self.model.name = model_name;
                self.model.init_value = [0.2, 1.0, -0.8];
                self.model.t = 0 : 0.05 : 250;
                self.model.args = [0.5, 2.0, 4.0];
                self.model.N = 400;
                self.model.rho = 1.0;
                self.model.sigma = 1.0;
                self.model.bias = 1.0;
                self.model.alpha = 1; %0.25
                self.model.beta = 1e-8;
                [~, y] = ode45(@(t, X) self.rossler(t, X, self.model.args), ...
                                       self.model.t, self.model.init_value);
            else
                error("Not only 'lorenz' and 'rossler' model are supported!")
            end
       
            y = (y-mean(y,1)) ./ mean((y - mean(y,1)).^2,1).^(1/2);
            self.model.states = y;
        end
    end
    
    methods(Access = protected)
        function dX = lorenz(self, t, X, args)
            a = args(1);
            b = args(2);
            c = args(3);
            dX = [-a*(X(1) - X(2));
                  b*X(1) - X(2) - X(1)*X(3);
                  X(1)*X(2) - c * X(3)];  
        end

        function dX = rossler(self, t, X, args)
            a = args(1);
            b = args(2);
            c = args(3);
            dX = [- X(2) - X(3);
                  X(1) + a*X(2);
                  b + X(3) * (X(1) - c)];            
        end
    end
    
    methods
        function disp(self)
           disp(self.model); 
        end
    end
end