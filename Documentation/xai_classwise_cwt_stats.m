function xai_classwise_cwt_stats()
% XAI-only analysis (no classifier):
% - Computes class-wise mean and std of CWT scalograms using streaming
% - Produces interpretable plots: mean/std heatmaps + band evidence curves
% - Works with huge X_cwt via matfile (no full load)
%
% Compatible with older MATLAB (no size(matfile,"var") syntax)

base  = "E:\Upwork Project\AI_Leak_Detection_Project\data\processed\cwt";
Xfile = fullfile(base, "X_cwt.mat");
yfile = fullfile(base, "y.mat");

classes = [ ...
    "Circumferential Crack"
    "Gasket Leak"
    "Longitudinal Crack"
    "No-leak"
    "Orifice Leak"
];

X = matfile(Xfile);
yy = load(yfile); y = int16(yy.y);

% ---- Get dimensions robustly (older MATLAB) ----
info = whos(X, "X_cwt");
if isempty(info)
    error("Variable X_cwt not found in %s", Xfile);
end
sz = info.size;              % [N F T] (or [N F T C] in general)
N = sz(1); F = sz(2); T = sz(3);
fprintf("[INFO] X_cwt size: N=%d, F=%d, T=%d\n", N, F, T);

K = numel(classes);

% ---- Streaming accumulators per class ----
sumS  = zeros(K, F, T, "single");
sumSq = zeros(K, F, T, "single");
count = zeros(K, 1);

% Stream in chunks
chunk = 200; % increase to 500 if RAM allows (faster)
for i = 1:chunk:N
    j = min(N, i+chunk-1);

    % Read chunk: (chunk x F x T)
    S = X.X_cwt(i:j, :, :);
    ychunk = y(i:j);

    for k = 0:K-1
        idx = find(ychunk == k);
        if ~isempty(idx)
            Sk = S(idx, :, :);                     % (#k x F x T)
            sumS(k+1,:,:)  = sumS(k+1,:,:)  + squeeze(sum(Sk,1));
            sumSq(k+1,:,:) = sumSq(k+1,:,:) + squeeze(sum(Sk.^2,1));
            count(k+1) = count(k+1) + numel(idx);
        end
    end

    if mod(i, chunk*10) == 1
        fprintf("[INFO] Processed %d/%d windows...\n", j, N);
    end
end

fprintf("[DONE] Counts per class:\n");
disp(table((0:K-1)', classes', count, 'VariableNames', {'label','class','count'}));

% ---- Mean and std ----
mu  = zeros(K, F, T, "single");
sig = zeros(K, F, T, "single");

for k = 1:K
    mu(k,:,:) = sumS(k,:,:) ./ max(1, count(k));
    v = (sumSq(k,:,:) ./ max(1, count(k))) - (mu(k,:,:).^2);
    v(v < 0) = 0;
    sig(k,:,:) = sqrt(v);
end

% ---- Save stats ----
outDir = fullfile(base, "xai_stats");
if ~exist(outDir, "dir"), mkdir(outDir); end
save(fullfile(outDir, "classwise_mu_sig.mat"), "mu", "sig", "count", "-v7.3");
fprintf("[INFO] Saved stats to %s\n", outDir);

% ---- Frequency vector (optional) ----
fPath = fullfile(base, "cwt_freq_vector.mat");
if exist(fPath, "file")
    tmp = load(fPath);
    f = tmp.f;
    if numel(f) ~= F
        warning("Saved frequency vector length != F. Using index axis instead.");
        f = 1:F;
    end
else
    f = 1:F; % fallback
end

% ---- Figures: mean and std ----
for k = 1:K
    M = squeeze(mu(k,:,:));     % F x T
    Sdev = squeeze(sig(k,:,:)); % F x T

    figure('Name',"Mean "+classes(k));
    imagesc(1:T, f, M); axis xy; colorbar
    xlabel("Time index"); ylabel("Frequency (Hz)");
    title("Class-wise mean scalogram: " + classes(k));
    saveas(gcf, fullfile(outDir, "mean_" + safe_name(classes(k)) + ".png"));

    figure('Name',"Std "+classes(k));
    imagesc(1:T, f, Sdev); axis xy; colorbar
    xlabel("Time index"); ylabel("Frequency (Hz)");
    title("Class-wise std scalogram: " + classes(k));
    saveas(gcf, fullfile(outDir, "std_" + safe_name(classes(k)) + ".png"));
end

% ---- Evidence curves (frequency signature) ----
E = zeros(K, F, "single");
for k = 1:K
    M = squeeze(mu(k,:,:));    % F x T
    E(k,:) = mean(M, 2);       % average over time
end

figure('Name','Class frequency evidence');
plot(f, E'); grid on
xlabel("Frequency (Hz)"); ylabel("Mean evidence (avg over time)");
title("XAI evidence signature per class (from mean scalograms)");
legend(classes, 'Location','best');
saveas(gcf, fullfile(outDir, "frequency_evidence_curves.png"));

% ---- Table: top evidence frequencies per class ----
topK = 8;
topTbl = table;
for k = 1:K
    [~, ix] = maxk(E(k,:), topK);
    topTbl.(safe_var(classes(k))) = f(ix)';
end
writetable(topTbl, fullfile(outDir, "top_freqs_per_class.csv"));
fprintf("[DONE] Wrote top frequency table.\n");

end

function s = safe_name(x)
s = regexprep(string(x), '[^\w]+', '_');
end

function v = safe_var(x)
v = matlab.lang.makeValidName(string(x));
end
