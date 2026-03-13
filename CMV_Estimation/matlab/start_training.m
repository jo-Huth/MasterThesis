tic;
% pyrunfile("./python_scripts/raft_finetuning_pseudolabel.py '--stage' '300'");
% pyrunfile("./python_scripts/raft_finetuning_pseudolabel.py '--stage' '600'");
% pyrunfile("./python_scripts/raft_finetuning_pseudolabel.py '--stage' '900'");
pyrunfile("./python_scripts/raft_finetuning_pseudolabel.py '--stage' '2000' '--epochs' '15'");
timeTraining = toc;