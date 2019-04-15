% Now only 'lorenz' and 'rossler' model are suported!
clear all;
close all;
lorenz = Model('lorenz');
r1 = Reservoir(lorenz)
r1.run();

rossler = Model('rossler');
r2 = Reservoir(rossler)
r2.run();
