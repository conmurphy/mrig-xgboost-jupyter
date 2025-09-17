import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import matplotlib.colors as mcolors

# Change default colours to custom colours
# Override SHAP default colors
col_shap_pos = "#22334D" # Change red to dark blue for positive SHAP values
col_shap_neg = "#3DA7CA" # Change blue to light blue for negative SHAP values
col_shap_new_vec = [col_shap_pos, col_shap_neg]
shap.plots.colors.red_rgb = mcolors.hex2color(col_shap_pos)  
shap.plots.colors.blue_rgb = mcolors.hex2color(col_shap_neg)

# Custom colormap
custom_cmap = mcolors.LinearSegmentedColormap.from_list("custom_colormap", col_shap_new_vec[::-1], N=100)

def plot_features (df,  target_col, discrete_threshold=10,max_cols=2, compare_by_target=1, percentiles=[0.1, 0.98]):
    '''
    Erzeugt KDE-Plots für metrische Merkmale, und Barplots für kategorische Merkmale und metrische Merkmale mit wenigen Werten.
    Parameter:
    df: Pandas DataFrame, der die Daten enthält.
    target_col: Name der binären Zielspalte (True/False).
    max_cols: Maximale Anzahl der Spalten im Plot-Raster (Standard ist 2)
    compare_by_target: (1/0) 1: KDE plots werden insgesamt normiert, sodass der Unterschied in Anzahl je target Variable erkennbar ist.
                                Bar plots werden nach Anzahlhöhe angezeigt und zeigen Prozentzahlen zum Erkennen der default prob.
                            0: KDE plots nach target normiert. Bar plots als Prozent je Target.
    percentiles: Liste mit zwei Werten zwischen 0 und 1, z.B. [0.01, 0.99].
    '''
    
    features = [col for col in df.columns if col != target_col]
    # Classify features
    numeric_feats = []
    categorical_feats = []

    for col in features:
        unique_vals = df[col].nunique()
        if pd.api.types.is_numeric_dtype(df[col]) and unique_vals > discrete_threshold:
            numeric_feats.append(col)
        else:
            categorical_feats.append(col)
    if (compare_by_target==1):
        plot_metric_features(df, numeric_feats, target_col, max_cols, percentiles, common_norm=True, alpha =1)
        plot_categorical_features_v2(df, categorical_feats, target_col,  max_cols)
        
    else:
        plot_metric_features(df, numeric_feats, target_col, max_cols, percentiles, common_norm=False, alpha = 0.5)
        plot_categorical_features(df, categorical_feats, target_col,  max_cols)
        
    


def plot_metric_features(df, features_col, target_col, max_cols=2, percentiles=None,common_norm=True,alpha=1):
    '''
    Erzeugt KDE-Plots für metrische Merkmale, gruppiert nach der binären Zielvariable.
    Die Plots werden in einem Raster mit einer angegebenen maximalen Anzahl von Spalten angeordnet.
    Parameter:
    df: Pandas DataFrame, der die Daten enthält.
    features_col: Liste der Spaltennamen der metrischen Merkmale, die geplottet werden sollen.
    target_col: Name der binären Zielspalte (True/False).
    max_cols: Maximale Anzahl der Spalten im Plot-Raster (Standard ist 2
    percentiles: Liste mit zwei Werten zwischen 0 und 1, z.B. [0.01, 0.99].
    common_norm: True/False. Bei True sind die KDE plots auf unterschiedlichen Höhen, bei False auf gleicher Höhe skaliert.
    Es werden nur die Werte im Bereich zwischen diesen Perzentilen auf der x-Achse angezeigt.
    '''
   
    # Schränke auf Merkmale ein, die im df enthalten sind
    features_col = [col for col in features_col if col in df.columns]

    # Konfiguriere das Plotting-Layout    
    total_plots = len(features_col) # Anzahl der zu plottenden Merkmale
    n_rows = int(np.ceil(total_plots / max_cols)) # Berechne die Anzahl der benötigten Zeilen
    fig, axes = plt.subplots(n_rows, max_cols, figsize=(6 * max_cols, 3 * n_rows),constrained_layout=True) 
    axes = axes.flatten() 
    fig.suptitle(f'Verteilungen metrischer Merkmale gruppiert nach {target_col}', fontsize=16) 
    plot_idx = 0
    for col in features_col:
        ax = axes[plot_idx]

        # Use KDE for continuous features
        g = sns.kdeplot(data=df, 
                        x=col, 
                        hue=target_col,
                        hue_order=[True, False], 
                        common_norm=common_norm, 
                        ax=ax, 
                        fill=True, 
                        alpha=alpha, 
                        palette=[col_shap_neg,col_shap_pos])

        # If tails are extreme apply clipping
        if percentiles is not None and len(percentiles) == 2:
            extra = (df[col].max() - df[col].min()) * 0.05

            lower = df[col].quantile(percentiles[0]) - extra
            upper = df[col].quantile(percentiles[1]) + extra
            q1, q3 = np.percentile(df[col].dropna(), [25, 75])
            # inter quartile range
            iqr = q3 - q1
            if (upper - lower) > 10 * iqr:
                lower = df[col].quantile(0.05) - extra
                upper = df[col].quantile(0.94) + extra
                ax.set_xlim(lower, upper)  
            elif (upper - lower) > 3 *iqr: 
                ax.set_xlim(lower, upper) 
        
        sns.move_legend(g, loc="upper right")
        ax.set_title(f"KDE: {col} by {target_col}")

        ax.set_xlabel("")
        plot_idx += 1

    
    # Entferne ungenutzte Achsen
    for j in range(plot_idx, len(axes)):
        fig.delaxes(axes[j])

    #plt.tight_layout(rect=[0, 0, 1, 0.98]) # Passe Layout an, um Überlappungen zu vermeiden
    plt.show()

def plot_categorical_features_v2(df, features_col, target_col, max_cols=2):
    '''
    Erzeugt Balkendiagramme für kategoriale Merkmale, gruppiert nach der binären Zielvariable
    mit Fokus auf Unterschied je Kategorie zwischen target_col Werten.
    Die Plots werden in einem Raster mit einer angegebenen maximalen Anzahl von Spalten angeordnet.
    Parameter:
    df: Pandas DataFrame, der die Daten enthält.
    features_col: Liste der Spaltennamen der kategorialen Merkmale, die geplottet werden sollen.
    target_col: Name der binären Zielspalte (True/False).
    max_cols: Maximale Anzahl der Spalten im Plot-Raster (Standard ist 2)    
    '''
   
    # Schränke auf Merkmale ein, die im df enthalten sind
    features_col = [col for col in features_col if col in df.columns]

    # Konfiguriere das Plotting-Layout    
    total_plots = len(features_col) # Anzahl der zu plottenden Merkmale
    n_rows = int(np.ceil(total_plots / max_cols)) # Berechne die Anzahl der benötigten Zeilen
    fig, axes = plt.subplots(n_rows, max_cols, figsize=(6 * max_cols, 4 * n_rows),constrained_layout=True) 
    axes = axes.flatten() 
    fig.suptitle(f'Verteilungen kategorialer (oder diskreter num.) Merkmale gruppiert nach {target_col}', fontsize=16) 
    plot_idx = 0
    
    # Erstelle Balkendiagramme
    for col in features_col:
        ax = axes[plot_idx]
        df_plot = df.copy()

        # Fehlende Werte mit "Missing" auffüllen
        df_plot[col] = df_plot[col].fillna("missing")

        # Kategorien mit weniger als 1 % zusammenfassen als "others"
        value_counts = df_plot[col].value_counts(normalize=True)
        rare_cats = value_counts[value_counts < 0.01].index
        df_plot[col] = df_plot[col].replace(rare_cats, "others")

        categories = sorted(df_plot[col].unique(), key=cat_sort_key)
        # Balkendiagramm mit absoluten Häufigkeiten zeichnen
        sns.countplot(data=df_plot, x=col, hue=target_col, ax=ax,palette=[col_shap_neg,col_shap_pos],order=categories)

        # Achsentitel und Layout anpassen
        ax.set_ylabel('Anzahl')
        ax.set_title(f"{col} vs {target_col}")
        ax.tick_params(axis='x', rotation=45)
        ax.set_xlabel("")

        # Gruppierung der Zieldaten zur Berechnung der Prozentwerte
        grouped = df_plot.groupby(col)[target_col].value_counts().unstack(fill_value=0)

        # Prozentwerte auf die Balken schreiben
        for container in ax.containers:
            for bar in container:
                height = bar.get_height()
                x_pos = bar.get_x() + bar.get_width() / 2

                # Passendes Kategorie-Label anhand der X-Achsen-Ticks ermitteln
                xtick_locs = ax.get_xticks()
                xtick_labels = [tick.get_text() for tick in ax.get_xticklabels()]
                idx = np.argmin(np.abs(xtick_locs - x_pos))
                category_label = xtick_labels[idx]

                try:
                    key = float(category_label)
                except ValueError:
                    key = category_label  # z.B. bei "Missing"
                # Gesamtanzahl innerhalb der Kategorie berechnen
                total = grouped.loc[key].sum()
                percent = height / total * 100 if total > 0 else 0

                # Prozentwert über dem Balken anzeigen
                ax.text(
                    x_pos, height,
                    f"{percent:.1f}%",
                    ha='center', va='bottom', fontsize=9
                )

        plot_idx += 1


    # Entferne ungenutzte Achsen
    for j in range(plot_idx, len(axes)):
        fig.delaxes(axes[j])

    #plt.tight_layout(rect=[0, 0, 1, 0.98]) # Passe Layout an, um Überlappungen zu vermeiden
    plt.show()

def cat_sort_key(x):
    # Missing always first
    if str(x).lower() == "missing":
        return (0, 0)
    # Numeric values in the middle
    try:
        return (1, float(x))
    except (ValueError, TypeError):
        pass
    # Strings (alphabetical), except "others"
    if str(x).lower() == "others":
        return (3, 0)   # push "others" to the very end
    return (2, str(x).lower())


def plot_categorical_features(df, features_col, target_col, max_cols=2):
    '''
    Erzeugt Balkendiagramme für kategoriale Merkmale, gruppiert nach der binären Zielvariable.
    Die Plots werden in einem Raster mit einer angegebenen maximalen Anzahl von Spalten angeordnet.
    Parameter:
    df: Pandas DataFrame, der die Daten enthält.
    features_col: Liste der Spaltennamen der kategorialen Merkmale, die geplottet werden sollen.
    target_col: Name der binären Zielspalte (True/False).
    max_cols: Maximale Anzahl der Spalten im Plot-Raster (Standard ist 2)    
    '''
   
    # Schränke auf Merkmale ein, die im df enthalten sind
    features_col = [col for col in features_col if col in df.columns]

    # Konfiguriere das Plotting-Layout    
    total_plots = len(features_col) # Anzahl der zu plottenden Merkmale
    n_rows = int(np.ceil(total_plots / max_cols)) # Berechne die Anzahl der benötigten Zeilen
    fig, axes = plt.subplots(n_rows, max_cols, figsize=(6 * max_cols, 5 * n_rows)) # Erstelle Subplots
    axes = axes.flatten() # Flache Liste der Achsen für einfaches Iterieren
    fig.suptitle(f'Verteilungen kategorialer Merkmale gruppiert nach {target_col}', fontsize=16) # Titel für die gesamte Figur
    plot_idx = 0
    
    # Erstelle Balkendiagramme
    for col in features_col:
        # Wähle die aktuelle Achse aus
        ax = axes[plot_idx]

        # Prozentuale Häufigkeiten berechnen 
        # - mit normalize='columns' für prozentuale Verteilung innerhalb jeder Zielkategorie 
        # - mit normalize='index' für prozentuale Verteilung innerhalb jeder Merkmalskategorie
        prop_df = (pd.crosstab(df[col], df[target_col], normalize='columns') * 100).reset_index().melt(id_vars=col, var_name=target_col, value_name='percent')

        # Balkendiagramm zeichnen
        sns.barplot(data=prop_df, x=col, y='percent', hue=target_col, ax=ax, palette=[col_shap_neg,col_shap_pos])
        ax.set_ylabel('Prozent')
        ax.set_title(f"{col} by {target_col}")
        ax.tick_params(axis='x', rotation=45)
        ax.set_xlabel("")

        # Prozentwerte auf die Balken schreiben
        for p in ax.patches:
            height = p.get_height()
            if height > 0:
                ax.text(
                    p.get_x() + p.get_width() / 2, height,
                    f"{height:.1f}%", ha="center", va="bottom", fontsize=9
                )
                
        plot_idx += 1

    # Entferne ungenutzte Achsen
    for j in range(plot_idx, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout(rect=[0, 0, 1, 0.98]) # Passe Layout an, um Überlappungen zu vermeiden
    plt.show()
