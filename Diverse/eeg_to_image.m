% Created on Thu Aug 18 15:27:15 2016
% @author: Albert Vilamala
% @email: alvmu@dtu.dk
% @affiliation: Technical University of Denmark
% @url: http://people.compute.dtu.dk/alvmu

% This code was used in the paper:
% Albert Vilamala, Kristoffer H. Madsen, Lars K. Hansen
% "Deep Convolutional Neural Networks for Interpretable Analysis of EEG Sleep Stage Scoring"
% which can be downloaded from: https://arxiv.org/abs/1710.00633

cd('PhysioNet_Sleep_EEG')
dirinfo = dir();
sensor = 'fpz';
offset = 2; %Number of files to skip in the directory

%Iterate over 39 files (19 subjects/2 nights and 1 subject/1 night)
for j=1+offset:39+offset 
    fprintf('Subject %d...\n', (j-offset))
    
    %Create directory where to store images
    mkdir(['sub' num2str(idivide(int16(j-offset-1),2)+1) '_n' num2str(mod(j,2)+1) '_img_' sensor])
    
    %Load hypnogram starting times
    fileID = fopen([dirinfo(j).name '/info/hyp_start_time.txt']);
    hyp_start = textscan(fileID,'%{HH:mm:ss.SSS}D');
    fclose(fileID);
    
    %Load lights off times
    fileID = fopen([dirinfo(j).name '/info/lights_off_time.txt']);
    lights_off = textscan(fileID,'%{HH:mm:ss}D');
    fclose(fileID);
    
    %Load lights on times
    fileID = fopen([dirinfo(j).name '/info/lights_on_time.txt']);
    lights_on = textscan(fileID,'%{HH:mm:ss.SSS}D');
    fclose(fileID);
    lights_on{1}=lights_on{1}+days(1);
    
    %Account for change of day
    if (lights_off{1} > datetime('00:00:00') && lights_off{1} < datetime('16:00:00'))
        lights_off{1} = lights_off{1} + days(1);
    end
    
    %Calculate sleep intervals
    pre_sleep_interval = seconds(lights_off{1}-hyp_start{1});
    sleep_interval = seconds(lights_on{1}-lights_off{1});
    
    %Load EEG sleep data
    filename = [dirinfo(j).name '/matlab/eeg_' sensor '.mat'];
    load(filename);
    
    %Load hynograms (labels)
    filename_hypno = [dirinfo(j).name '/matlab/hypnogram.mat'];
    load(filename_hypno)
    hypnogram = hypnogram(pre_sleep_interval/30 + 1:pre_sleep_interval/30 + sleep_interval/30);
    dlmwrite(['sub' num2str(idivide(int16(j-offset-1),2)+1) '_n' num2str(mod(j,2)+1) '_img_' sensor '/labels.txt'],hypnogram)
    
    %Calculate sizes
    num_epochs = length(hypnogram);
    samples_x_epoch = 3000;
    frequency = 100;
    
    %Select signal intervals
    signal = signal(pre_sleep_interval*frequency + 1:pre_sleep_interval*frequency + sleep_interval*frequency);
    
    %Set MultiTaper Spectrogram Params
    window = 3;
    movingwin = 0.67;  %0.13; % to get a 224 resolution at the temporal axis
    freq_res = 2;
    tw = (window * freq_res)/2;
    num_tapers = floor(2*tw)-1;
    params = struct('Fs',frequency,'tapers',[tw num_tapers],'fpass',[0 30]);
    
    %Calcualte spectrograms
    [S,t,f]=mtspecgramc(signal,[window movingwin],params);
    S=S';
    S=flipud(S);
    S=log10(S+1.0);
    
    %Threshold spectrogram amplitude values
    max_val = 1.0; %0.85*max(S(:)); % mean(S(:))+std(S(:));%0.85*max(S(:));
    min_val = 0.0; %min(S(:));
    C = colormap(jet(255));  % Get the figure's colormap.
    L = size(C,1);    
    
    % Print the whole night image
    % Scale the matrix to the range of the map.
    S = histeq(S,255);
    currentS=S;
    
    %Print whole night spectrogram image
    Gs = round(interp1(linspace(min_val,max_val,L),1:L,currentS));
    H = reshape(C(Gs,:),[size(Gs) 3]); % Make RGB image from scaled.
    imwrite(H,['sub' num2str(idivide(int16(j-offset-1),2)+1) '_n' num2str(mod(j,2)+1) '_img_' sensor '/img_whole_night.png']);
    
    %Write parameters
    fid=fopen(['sub' num2str(idivide(int16(j-offset-1),2)+1) '_n' num2str(mod(j,2)+1) '_img_' sensor '/params.txt'],'w');
    fprintf(fid, 'window: %d\nmoving_win: %f\nfreq_res: %d\nmax_val: %f\nmin_val: %f\n', window, movingwin,freq_res, max_val, min_val);
    fclose(fid);
    
    %Split the whole night into epoched images
    for ep=1:num_epochs
        currentS = S(:,t>30*(ep-1)-60 & t<30*ep+60);       
        
        % Scale the matrix to the range of the map.       
        currentS(currentS>max_val)=max_val;
        currentS(currentS<min_val)=min_val;
        
        %Print epoched image
        Gs = round(interp1(linspace(min_val,max_val,L),1:L,currentS));
        H = reshape(C(Gs,:),[size(Gs) 3]); % Make RGB image from scaled.
        H = imresize(H,[224 224]);
        imwrite(H,['sub' num2str(idivide(int16(j-offset-1),2)+1) '_n' num2str(mod(j,2)+1) '_img_' sensor '/img_' num2str(ep) '.png']);
    end

end

disp('EEG spectrogram images have been created!')
