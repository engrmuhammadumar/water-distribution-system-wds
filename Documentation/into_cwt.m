function into_cwt()
% into_cwt.m
% Robust XAI-first CWT preprocessing for leak dataset (Accelerometer/Branched)
%
% What it does:
% - Reads CSV signals from class folders
% - Per-file z-score normalization (XAI-safe)
% - Sliding windows: win=4096, overlap=50% (stride=2048)
% - Computes CWT scalograms per window (log1p(abs(wt)))
% - Saves using matfile (supports large arrays)
% - Writes meta.csv and class_map.json
%
% Key robustness:
% - Works when data is single-channel (C=1)
% - Avoids trailing singleton dimension issues by storing:
%     * C==1 -> X_cwt is 3D: (N,F,T)
%     * C>1  -> X_cwt is 4D: (N,F,T,C)
% - matfile assignment uses reshape to match slice dims exactly
% - Metadata writing uses string arrays (no strjoin error)

% ===================== PATHS (EDIT IF NEEDED) =====================
inputRoot  = "E:\Upwork Project\AI_Leak_Detection_Project\data\raw\Accelerometer\Branched";
outputRoot = "E:\Upwork Project\AI_Leak_Detection_Project\data\processed";

% ===================== SETTINGS =====================
fs      = 51200;    % accelerometer sampling rate (Hz)
win     = 4096;     % samples (~80 ms)
overlap = 0.50;
stride  = round(win*(1-overlap));

% CWT parameters
voicesPerOctave = 12;
freqLimits      = [50 20000];   % Hz (adjust if you want)
useAbsLog1p      = true;        % log1p(|CWT|) for better dynamic range

classes = [ ...
    "Circumferential Crack"
    "Gasket Leak"
    "Longitudinal Crack"
    "No-leak"
    "Orifice Leak"
];

% ===================== OUTPUT =====================
cwtDir = fullfile(outputRoot, "cwt");
if ~exist(cwtDir, "dir"), mkdir(cwtDir); end

X_cwt_path = fullfile(cwtDir, "X_cwt.mat");
y_path     = fullfile(cwtDir, "y.mat");
meta_path  = fullfile(cwtDir, "meta.csv");
classmap_path = fullfile(outputRoot, "class_map.json");

% Clean previous outputs (prevents shape conflicts)
if exist(X_cwt_path, "file"), delete(X_cwt_path); end
if exist(y_path, "file"), delete(y_path); end
if exist(meta_path, "file"), delete(meta_path); end

% ===================== CLASS MAP =====================
classMap = containers.Map(classes, num2cell(0:numel(classes)-1));
write_class_map_json(classmap_path, classes);

% ===================== ENUMERATE FILES =====================
fileList  = {};
fileClass = {};
for ci = 1:numel(classes)
    clsDir = fullfile(inputRoot, classes(ci));
    if ~exist(clsDir, "dir")
        error("Missing class folder: %s", clsDir);
    end
    files = dir(fullfile(clsDir, "*.csv"));
    for k = 1:numel(files)
        fileList{end+1,1}  = fullfile(files(k).folder, files(k).name); %#ok<AGROW>
        fileClass{end+1,1} = classes(ci); %#ok<AGROW>
    end
end

nFiles = numel(fileList);
fprintf("[INFO] Found %d CSV files.\n", nFiles);
if nFiles == 0
    error("No CSV files found under: %s", inputRoot);
end

% ===================== PASS 1: CHANNELS + TOTAL WINDOWS =====================
totalWindows = 0;
C = [];

for i = 1:nFiles
    sig = read_signal_csv(fileList{i});       % NxC0 (1..3)
    sig = per_file_zscore(sig);
    sig = add_magnitude_if_3axis(sig);        % NxC (adds only if 3-axis)

    if isempty(C)
        C = size(sig,2);
    elseif size(sig,2) ~= C
        error("Channel mismatch: first file C=%d but %s has C=%d", C, fileList{i}, size(sig,2));
    end

    totalWindows = totalWindows + count_windows(size(sig,1), win, stride);
end

fprintf("[INFO] Channels C=%d. Total windows N=%d.\n", C, totalWindows);
if totalWindows == 0
    error("No windows formed. Check win/stride or file lengths.");
end

% ===================== PROBE CWT SHAPE =====================
probeDone = false;
Fbins = []; Tbins = [];

for i = 1:nFiles
    sig = add_magnitude_if_3axis(per_file_zscore(read_signal_csv(fileList{i})));
    if size(sig,1) >= win
        x = sig(1:win, 1);
        [wt, ~] = cwt(x, fs, "VoicesPerOctave", voicesPerOctave, "FrequencyLimits", freqLimits);
        S = abs(wt);
        if useAbsLog1p, S = log1p(S); end
        Fbins = size(S,1);
        Tbins = size(S,2);
        probeDone = true;
        break;
    end
end

if ~probeDone
    error("Could not probe CWT shape (no file long enough for win=%d).", win);
end

fprintf("[INFO] CWT shape per window: F=%d, T=%d, C=%d\n", Fbins, Tbins, C);

% ===================== PREALLOCATE MATFILES =====================
mX = matfile(X_cwt_path, "Writable", true);
mY = matfile(y_path, "Writable", true);

% Store as 3D if C==1, else 4D
if C == 1
    mX.X_cwt = zeros(totalWindows, Fbins, Tbins, "single");        % (N,F,T)
else
    mX.X_cwt = zeros(totalWindows, Fbins, Tbins, C, "single");     % (N,F,T,C)
end

mY.y = zeros(totalWindows, 1, "int16");

% ===================== METADATA =====================
metaHeader = ["global_index","class_name","label","file_path","file_id","fs_hz", ...
              "window_start","window_end","win_samples","stride_samples","channels"];
write_meta_header(meta_path, metaHeader);

% ===================== PASS 2: PROCESS + SAVE =====================
g = 1;

for file_id = 1:nFiles
    cls   = fileClass{file_id};
    label = int16(classMap(cls));
    fpath = fileList{file_id};

    sig = add_magnitude_if_3axis(per_file_zscore(read_signal_csv(fpath)));
    N = size(sig,1);
    nwin = count_windows(N, win, stride);

    if nwin == 0
        fprintf("[WARN] Skipping short file: %s\n", fpath);
        continue;
    end

    for wi = 0:(nwin-1)
        s = wi*stride + 1;
        e = s + win - 1;
        window = sig(s:e, :); % win x C

        if C == 1
            [wt, ~] = cwt(window(:,1), fs, "VoicesPerOctave", voicesPerOctave, "FrequencyLimits", freqLimits);
            S = abs(wt);
            if useAbsLog1p, S = log1p(S); end
            S = force_shape(S, Fbins, Tbins);   % single(F x T)

            % matfile slice is 1xF xT, so reshape RHS to 1xF xT
            mX.X_cwt(g, :, :) = reshape(S, 1, Fbins, Tbins);

        else
            for ch = 1:C
                [wt, ~] = cwt(window(:,ch), fs, "VoicesPerOctave", voicesPerOctave, "FrequencyLimits", freqLimits);
                S = abs(wt);
                if useAbsLog1p, S = log1p(S); end
                S = force_shape(S, Fbins, Tbins);

                % matfile slice is 1xF xT x1, so reshape RHS to 1xF xT x1
                mX.X_cwt(g, :, :, ch) = reshape(S, 1, Fbins, Tbins, 1);
            end
        end

        mY.y(g, 1) = label;

        row = {g, cls, double(label), fpath, file_id, fs, s, e, win, stride, C};
        append_meta_row(meta_path, row);

        g = g + 1;
    end

    fprintf("[INFO] %d/%d processed: %s -> %d windows\n", file_id, nFiles, string(fpath), nwin);
end

fprintf("[DONE] Filled windows: %d (preallocated %d)\n", g-1, totalWindows);
fprintf("Saved: %s\n", cwtDir);
fprintf("Meta:  %s\n", meta_path);
fprintf("Map:   %s\n", classmap_path);

end

% ===================== HELPERS =====================

function sig = read_signal_csv(fpath)
% Reads numeric signal columns robustly from CSV:
% - keeps numeric columns
% - drops likely monotonic time/index column if present
% - keeps up to first 3 columns (ax,ay,az) if multiple exist

opts = detectImportOptions(fpath, "NumHeaderLines", 0);
T = readtable(fpath, opts);

% numeric columns
numVars = varfun(@isnumeric, T, "OutputFormat","uniform");
Tn = T(:, numVars);

% if none numeric, coerce strings to numeric
if width(Tn) == 0
    T2 = T;
    for j = 1:width(T2)
        if ~isnumeric(T2{:,j})
            x = str2double(string(T2{:,j}));
            if any(~isnan(x))
                T2{:,j} = x;
            end
        end
    end
    numVars2 = varfun(@isnumeric, T2, "OutputFormat","uniform");
    Tn = T2(:, numVars2);
end

if width(Tn) == 0
    error("No numeric columns found in: %s", fpath);
end

X = single(table2array(Tn));

% Drop likely time/index column if monotonic and other cols exist
if size(X,2) >= 2
    drop = false(1, size(X,2));
    for c = 1:size(X,2)
        x = X(1:min(end,20000), c);
        dx = diff(x);
        if (all(dx >= 0) || all(dx <= 0)) && std(x) > 0
            drop(c) = true;
        end
    end
    if any(drop) && sum(~drop) >= 1
        X = X(:, ~drop);
    end
end

% keep at most first 3 channels
X = X(:, 1:min(3, size(X,2)));

% clean NaN/Inf
for c = 1:size(X,2)
    x = X(:,c);
    bad = ~isfinite(x);
    if any(bad)
        idx = (1:numel(x))';
        good = ~bad;
        if nnz(good) >= 2
            x(bad) = interp1(idx(good), x(good), idx(bad), "linear", "extrap");
        else
            x(bad) = 0;
        end
        X(:,c) = x;
    end
end

sig = X;
end

function sig = per_file_zscore(sig)
mu = mean(sig, 1, "omitnan");
sd = std(sig, 0, 1, "omitnan");
sig = (sig - mu) ./ (sd + 1e-8);
end

function sig2 = add_magnitude_if_3axis(sig)
if size(sig,2) == 3
    mag = sqrt(sum(sig.^2, 2));
    sig2 = [sig, mag];
else
    sig2 = sig;
end
end

function nwin = count_windows(N, win, stride)
if N < win
    nwin = 0;
else
    nwin = 1 + floor((N - win) / stride);
end
end

function S = force_shape(S, Fbins, Tbins)
% Ensure single(Fbins x Tbins) by pad/crop if needed
S = single(S);
if size(S,1) ~= Fbins || size(S,2) ~= Tbins
    S2 = zeros(Fbins, Tbins, "single");
    fmin = min(Fbins, size(S,1));
    tmin = min(Tbins, size(S,2));
    S2(1:fmin, 1:tmin) = S(1:fmin, 1:tmin);
    S = S2;
end
end

function write_meta_header(csvPath, header)
fid = fopen(csvPath, "w");
if fid < 0, error("Cannot write: %s", csvPath); end
fprintf(fid, "%s\n", strjoin(string(header), ",")); % force string array
fclose(fid);
end

function append_meta_row(csvPath, row)
fid = fopen(csvPath, "a");
if fid < 0, error("Cannot append: %s", csvPath); end

out = strings(1, numel(row));   % IMPORTANT: string array (not cell)
for i = 1:numel(row)
    v = row{i};
    if isstring(v) || ischar(v)
        s = string(v);
        s = strrep(s, """", """""");   % escape quotes
        out(i) = """" + s + """";
    else
        out(i) = string(v);
    end
end

fprintf(fid, "%s\n", strjoin(out, ","));
fclose(fid);
end

function write_class_map_json(outPath, classes)
S = struct();
for i = 1:numel(classes)
    key = matlab.lang.makeValidName(classes(i));
    S.(key) = i-1;
end
txt = jsonencode(S);
fid = fopen(outPath, "w");
if fid < 0, error("Cannot write: %s", outPath); end
fwrite(fid, txt, "char");
fclose(fid);
end
