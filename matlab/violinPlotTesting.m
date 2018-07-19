clearvars
close all
%get probability distribution
load('exp_results_on_noise10.mat')
probs = double(p_arrays(1,:)');

%sample N points using MC method to turn probability distribution into
%set of simulated data predictions
tot_prob = sum(probs);
partitions = [0; cumsum(probs)]./tot_prob;
N = 1e5;
list = zeros(N,1);
tic
for i = 1:N
   num = rand;
   event = find(sort([partitions; num]) == num)-2;
   list(i) = event;
end
toc
plot(0:51,probs)
xlim([0,52])
figure;
hist(list,52)
xlim([0,52])
