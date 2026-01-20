clear; clc; close all;

%% ------------------ PATHS ------------------
DATA_ROOT = "E:\Collaboration Work\With Farooq\phm dataset\PHM Challange 2010 Milling";
cutter = "c1";                              % c1/c4/c6
cutFolder = fullfile(DATA_ROOT, cutter, cutter);

%% ------------------ SETTINGS ------------------
FS  = 50000;
WIN = 4096;
HOP = 2048;

SENSOR_NAMES = ["Force_X","Force_Y","Force_Z", ...
                "Vibration_X","Vibration_Y","Vibration_Z","AE_RMS"];

% choose which sensors to use for feature extraction
USE_SENSORS = SENSOR_NAMES;   % or e.g. ["Force_X","Force_Y","Force_Z"]

% features to compute per window per sensor (simple + common)
% (you can add more later)
featureNames = ["mean","rms","std","skew","kurtosis","ptp","spec_centroid","bandpower"];
nFeatPerSensor = numel(featureNames);

%% ------------------ LIST CUT FILES ------------------
files = dir(fullfile(cutFolder, "c_*.csv"));
if isempty(files)
    error("No cut files found in: %s", cutFolder);
end

% sort by cut index
cutIdxAll = zeros(numel(files),1);
for i=1:numel(files)
    % filename example: c_1_001.csv -> take last 3 digits
    name = files(i).name;
    cutIdxAll(i) = str2double(extractBefore(extractAfter(name, "_"), ".csv"));
end
[cutIdxAll, order] = sort(cutIdxAll);
files = files(order);

%% ------------------ PREP FEATURE MATRIX ------------------
nCuts = numel(files);

% sensor indices
useIdx = find(ismember(SENSOR_NAMES, USE_SENSORS));
nSensors = numel(useIdx);

% total features per cut
Fdim = nSensors * nFeatPerSensor;

F = zeros(nCuts, Fdim);     % features per cut
F_labels = strings(1, Fdim);

% build labels
p = 1;
for s = 1:nSensors
    for f = 1:nFeatPerSensor
        F_labels(p) = USE_SENSORS(s) + "_" + featureNames(f);
        p = p + 1;
    end
end

%% ------------------ MAIN LOOP: FEATURE EXTRACTION ------------------
for c = 1:nCuts
    X = readmatrix(fullfile(files(c).folder, files(c).name));
    X = X(:,1:7); % ensure 7 columns

    X = X(:,useIdx);   % keep selected sensors
    N = size(X,1);

    % z-score per sensor (recommended so features comparable)
    Xz = zscore(X);

    if N < WIN
        % pad to one window
        pad = WIN - N;
        Xz = [Xz; repmat(Xz(end,:), pad, 1)];
        N = size(Xz,1);
    end

    starts = 1:HOP:(N - WIN + 1);
    nW = numel(starts);

    % window features -> then aggregate (mean across windows)
    winFeat = zeros(nW, Fdim);

    for w = 1:nW
        s0 = starts(w);
        seg = Xz(s0:s0+WIN-1, :);  % WIN x nSensors

        featVec = zeros(1, Fdim);
        k = 1;

        for si = 1:nSensors
            x = seg(:,si);

            % time-domain
            mu  = mean(x);
            rr  = rms(x);
            sd  = std(x);
            sk  = skewness(x);
            ku  = kurtosis(x);
            ptp = peak2peak(x);

            % freq-domain (simple)
            Xf = abs(fft(x));
            Xf = Xf(1:floor(end/2));                 % one-sided magnitude
            faxis = (0:numel(Xf)-1)' * (FS/WIN);

            spec_centroid = sum(faxis .* Xf) / (sum(Xf) + 1e-12);

            % bandpower (0..Nyquist)
            bandpower_all = sum(Xf.^2);              % proxy power (no scaling)

            vals = [mu, rr, sd, sk, ku, ptp, spec_centroid, bandpower_all];

            featVec(k:k+nFeatPerSensor-1) = vals;
            k = k + nFeatPerSensor;
        end

        winFeat(w,:) = featVec;
    end

    % aggregate windows -> one vector per cut
    F(c,:) = mean(winFeat, 1);
end

%% ------------------ NORMALIZE FEATURES TO [0,1] FOR PLOTTING ------------------
Fmin = min(F, [], 1);
Fmax = max(F, [], 1);
Fnorm = (F - Fmin) ./ (Fmax - Fmin + 1e-12);

%% ------------------ PLOT LIKE YOUR IMAGE ------------------
figure('Color','w','Position',[80 80 1100 520]);
hold on; box on;

for j = 1:size(Fnorm,2)
    plot(cutIdxAll, Fnorm(:,j), 'LineWidth', 1.0);
end

title("Extracted features", 'FontWeight','bold');
xlabel("Cut index", 'FontWeight','bold');
ylabel("Normalized feature value (0–1)", 'FontWeight','bold');
set(gca,'FontWeight','bold','LineWidth',1.2,'FontSize',12);
grid on; set(gca,'GridAlpha',0.15);
xlim([min(cutIdxAll) max(cutIdxAll)]);
ylim([0 1]);
hold off;

%% (Optional) If too many curves, you can plot only top-K most varying
% varF = var(Fnorm, 0, 1);
% [~,idx] = sort(varF, 'descend');
% K = 60; idx = idx(1:K);
% figure('Color','w'); hold on; box on;
% for j = idx
%     plot(cutIdxAll, Fnorm(:,j), 'LineWidth', 1.0);
% end
% title("Extracted features (top varying)",'FontWeight','bold');
% xlabel("Cut index",'FontWeight','bold');
% ylabel("Normalized feature value (0–1)",'FontWeight','bold');
% set(gca,'FontWeight','bold','LineWidth',1.2,'FontSize',12);
% grid on; set(gca,'GridAlpha',0.15);
% ylim([0 1]); hold off;
