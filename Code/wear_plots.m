%% ================================================================
%  WEAR CURVES – MATCH PYTHON STYLE EXACTLY (MATLAB)
%% ================================================================

clear; clc; close all;

DATA_ROOT = "E:\Collaboration Work\With Farooq\phm dataset\PHM Challange 2010 Milling";
SAVE_DIR  = "E:\4 Paper\4 Paper\Results\final_final_diagrams\wear_plots";
if ~exist(SAVE_DIR,"dir"), mkdir(SAVE_DIR); end

cutters_to_plot = ["c1","c4","c6"];

for c = 1:numel(cutters_to_plot)
    cutter = cutters_to_plot(c);

    wear_file = fullfile(DATA_ROOT, cutter, cutter + "_wear.csv");
    if ~isfile(wear_file)
        warning("Missing file: %s", wear_file);
        continue;
    end

    df = readtable(wear_file);

    % Cut index (PHM safe)
    if ismember("cut", df.Properties.VariableNames)
        cut_idx = df.cut;
    else
        cut_idx = (1:height(df))';
    end

    % ------------------------------------------------------------
    % FIGURE SIZE — SAME AS PYTHON (8 x 5 inches)
    % ------------------------------------------------------------
    fig = figure( ...
        "Color","w", ...
        "Units","pixels", ...
        "Position",[200 200 800 500]);

    ax = axes(fig); hold(ax,"on");

    % Thin lines + small markers (same visual weight as Python)
    plot(ax, cut_idx, df.flute_1, "-o", ...
        "LineWidth",0.8, "MarkerSize",4, "DisplayName","Flute 1");
    plot(ax, cut_idx, df.flute_2, "-s", ...
        "LineWidth",0.8, "MarkerSize",4, "DisplayName","Flute 2");
    plot(ax, cut_idx, df.flute_3, "-^", ...
        "LineWidth",0.8, "MarkerSize",4, "DisplayName","Flute 3");

    % Labels (bold, same scale as Python)
    xlabel(ax,"Cut Number","FontWeight","bold","FontSize",11);
    ylabel(ax,"Tool Wear (\mum)","FontWeight","bold","FontSize",11);
    title(ax, upper(cutter),"FontWeight","bold","FontSize",12);

    % ------------------------------------------------------------
    % GRID — SAME AS PYTHON
    % ------------------------------------------------------------
    grid(ax,"on");
    ax.GridAlpha = 0.30;
    ax.MinorGridAlpha = 0.30;
    ax.GridLineStyle = "-";
    ax.MinorGridLineStyle = "-";

    % ------------------------------------------------------------
    % BORDERS — SAME AS PYTHON (DEFAULT, NOT THICK)
    % ------------------------------------------------------------
    box(ax,"on");
    ax.LineWidth = 0.8;

    % ------------------------------------------------------------
    % TICKS — BOLD BUT NOT LARGE
    % ------------------------------------------------------------
    ax.FontSize = 10;
    ax.FontWeight = "bold";
    ax.XAxis.FontWeight = "bold";
    ax.YAxis.FontWeight = "bold";

    % ------------------------------------------------------------
    % LEGEND — BOLD (MATCH REQUEST)
    % ------------------------------------------------------------
    leg = legend(ax,"Location","best");
    leg.FontSize = 10;
    leg.FontWeight = "bold";
    leg.Box = "off";

    % ------------------------------------------------------------
    % SAVE — SAME QUALITY AS PYTHON (600 DPI)
    % ------------------------------------------------------------
    out_file = fullfile(SAVE_DIR, cutter + "_wear_plot.png");
    exportgraphics(fig, out_file, "Resolution",600);

    fprintf("✅ Saved: %s\n", out_file);
end

fprintf("\n✔ All wear plots saved to:\n%s\n", SAVE_DIR);
