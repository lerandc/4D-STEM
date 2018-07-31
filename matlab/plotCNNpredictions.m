%%
%set up
clearvars
close all
colors = lines(8);
colors(end,:) = [1 0.7333 1];
colors = [colors; 148/255,148/255,148/255];
colors = [colors; 0,100/255,0];
averages = false;
files = dir('*.mat');
prob_cutoff = 0.25;
use_cutoff = true;
use_npeaks = false;

for file = {files.name}
    %% prepare data
    file_name = file{1};
    graph_name = strrep(file_name(1:end-4),'_','-');
    load(file_name)
    
    if use_cutoff
        indices = max(probabilities,[],2) > prob_cutoff;
    elseif use_npeaks
        npeaks = countPeaks(probabilities);
        indices = countPeaks(probabilities) < 6;
    else
        indices = logical(ones(size(probabilities,1),1));
    end
    
    sets = sets(indices,:);
    radii = radii(indices,:);
    
    %convert layer predictions to nm
    if contains(file_name,'resnet')
        [~,predicted] = max(probabilities,[],2);
        predicted = (double(predicted'))*1.9525;
        measured = measured;
        u_sets = unique(sets,'rows');
        u_rad = unique(radii,'rows');
    else
        if averages
            %a = sum(probabilities,2);
            %scale_factor = 1./a;
            norm_prob = probabilities;%.*scale_factor;
            b = sum(norm_prob,2);
            c = .3905*(1:91);
            heights = 1.9525*(1:52);
            weights = norm_prob.*heights;
            predicted = sum(weights,2);
        else
            predicted = (double(predicted(indices)')+1)*.3905;%1.9525;
            %predicted = (double(probabilities)+1)*.3905;
        end
        measured = measured(indices)';
        u_sets = unique(sets,'rows');
        u_rad = unique(radii,'rows');
    end
    %make filters for each unique radius, set
    s_filter = [];
    for set = cellstr(u_sets)'
        s_filter = [s_filter strcmp(cellstr(sets),set)];
    end

    r_filter = [];
    for rad = cellstr(u_rad)'
        r_filter = [r_filter strcmp(cellstr(radii),rad)];
    end

    %% plot all such things
    fig1 = figure;
    hold on
    title(['predictions by set, ', graph_name])
    fig2 = figure;
    hold on
    title(['predictions by integration radius, ',graph_name])


    %by set
    figure(fig1)
    for i = 1:size(s_filter,2)
        filter = logical(s_filter(:,i));
        x = measured(filter);
        y = predicted(filter);
        radius = unique(radii(filter,:),'rows');
        set = strtrim(unique(sets(filter,:),'rows'));
        scatter(x,y,[],getColorS(colors,set),'Filled','MarkerEdgeColor','k',...
            'MarkerFaceAlpha',0.7)
    end
    plot(1:100,1:100,'k--')
    grid on
    ylabel('Predicted thickness (nm)')
    xlabel('HAADF measured thickness (nm)')
    h = zeros(7, 1);
    for i = 1:10
        h(i) = plot(NaN,NaN,'o','MarkerFaceColor',colors(i,:),'MarkerEdgeColor','k');
    end
    legend(h, 'S1','S2','S3','S4','S5','S6','S7','S8','S9','S10','Location','southeast');

    figure(fig2)
    %by integration radius
    for i = 1:size(r_filter,2)
        filter = logical(r_filter(:,i));
        x = measured(filter);
        y = predicted(filter);
        radius = unique(radii(filter,:),'rows');
        set = strtrim(unique(sets(filter,:),'rows'));
        scatter(x,y,[],colors(1,:),'Filled','MarkerEdgeColor','k',...
            'MarkerFaceAlpha',0.7)
        %scatter(x,y,[],getColorR(colors,radius),'Filled','MarkerEdgeColor','k',...
            %'MarkerFaceAlpha',0.7)
    end
    plot(1:100,1:100,'k--')
    grid on
    ylabel('Predicted thickness (nm)')
    xlabel('HAADF measured thickness (nm)')

    h = zeros(8, 1);
    for i = 1:8
        h(i) = plot(NaN,NaN,'o','MarkerFaceColor',colors(i,:),'MarkerEdgeColor','k');
    end
    legend(h,'R1','R2','R3','R4','R5','R6','R7','R8','Location','southeast');

    savefig(fig1,strcat(file_name(1:end-4),'_fig1.fig'))
    %savefig(fig2,strcat(file_name(1:end-4),'_fig2.fig'))
    close all
    clearvars -except files colors file averages prob_cutoff use_cutoff use_npeaks
end
    
%% functions

function [color] = getColorS(colors,identifier)
    switch identifier
        case 'S1'
            color = colors(1,:);
        case 'S2'
            color = colors(2,:);
        case 'S3'
            color = colors(3,:);
        case 'S4'
            color = colors(4,:);
        case 'S5'
            color = colors(5,:);
        case 'S6'
            color = colors(6,:);
        case 'S7'
            color = colors(7,:);
        case 'S8'
            color = colors(8,:);
        case 'S9'
            color = colors(9,:);
        case 'S10'
            color = colors(10,:);
    end
end

function [color] = getColorR(colors,identifier)
    switch identifier
        case 'R1'
            color = colors(1,:);
        case 'R2'
            color = colors(2,:);
        case 'R3'
            color = colors(3,:);
        case 'R4'
            color = colors(4,:);
        case 'R5'
            color = colors(5,:);
        case 'R6'
            color = colors(6,:);
        case 'R7'
            color = colors(7,:);
        case 'R8'
            color = colors(8,:);
    end
end

function nPeaks = countPeaks(probabilities)
    nPeaks = zeros(size(probabilities,1),1);
    for i = 1:length(nPeaks)
        nPeaks(i) = size(findpeaks(probabilities(i,:),'MinPeakDistance',4),2);
    end
end