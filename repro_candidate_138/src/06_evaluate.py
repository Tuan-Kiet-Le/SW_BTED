"""
Stage 6: Main Evaluation & Report Generator
This script evaluates the SW-BTED algorithm against labeled pairs, generates plots, and prints the summary.
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate():
    print("Generating Scientific Evaluation Plots and Reports...")
    
    # Load results
    metrics_path = "results/evaluation_metrics.csv"
    pairs_path = "results/pair_similarities.csv"
    runtime_path = "results/runtime_benchmark.csv"
    stat_path = "results/statistical_tests.csv"
    
    if not (os.path.exists(metrics_path) and os.path.exists(pairs_path) and os.path.exists(runtime_path)):
        print("Error: Missing result files in 'results/'. Please run main_evaluation and runtime_benchmark first.")
        return
        
    df_metrics = pd.read_csv(metrics_path)
    df_pairs = pd.read_csv(pairs_path)
    df_runtime = pd.read_csv(runtime_path)
    df_stat = pd.read_csv(stat_path) if os.path.exists(stat_path) else None
    
    # Set style
    sns.set_theme(style="whitegrid")
    
    # --- Plot 1: F1-Score Comparison with Error Bars ---
    plt.figure(figsize=(10, 6))
    
    methods = df_metrics["Method"].tolist()
    f1_means = df_metrics["F1_Score_Mean"].tolist()
    f1_stds = df_metrics["F1_Score_Std"].fillna(0).tolist()
    
    colors = sns.color_palette("coolwarm", len(methods))
    bars = plt.bar(methods, f1_means, yerr=f1_stds, align='center', alpha=0.8, ecolor='black', capsize=10, color=colors, edgecolor='black')
    
    plt.title("F1-Score Comparison with Cross-Validation Error Bars", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Method", fontsize=12)
    plt.ylabel("F1-Score (Mean ± Std)", fontsize=12)
    plt.ylim(0, 1.15)
    plt.xticks(rotation=15, ha='right')
    
    # Add values on top of bars
    for bar, mean, std in zip(bars, f1_means, f1_stds):
        height = bar.get_height()
        plt.annotate(f"{mean:.4f}\n(±{std:.4f})",
                    (bar.get_x() + bar.get_width() / 2., height + 0.02),
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
                    
    plt.tight_layout()
    plt.savefig("results/performance_comparison.png", dpi=300)
    plt.close()
    print("Saved performance comparison plot to results/performance_comparison.png")
    
    # --- Plot 2: Runtime Scaling ---
    plt.figure(figsize=(8, 5))
    sns.lineplot(data=df_runtime, x="Database_Size_N", y="Total_Time_Seconds", marker="o", color="blue", linewidth=2.5)
    plt.title("Query Execution Time Scaling", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Database Size N", fontsize=12)
    plt.ylabel("Total Query Time (seconds)", fontsize=12)
    
    # Annotate points
    for idx, row in df_runtime.iterrows():
        plt.annotate(f"{row['Total_Time_Seconds']:.2f}s\n({row['Avg_Time_Per_Comparison_MS']:.1f}ms/pair)",
                     (row['Database_Size_N'], row['Total_Time_Seconds']),
                     textcoords="offset points", xytext=(0,10), ha='center', fontsize=9,
                     bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.3))
                     
    plt.ylim(0, df_runtime["Total_Time_Seconds"].max() * 1.3)
    plt.tight_layout()
    plt.savefig("results/runtime_scaling.png", dpi=300)
    plt.close()
    print("Saved runtime scaling plot to results/runtime_scaling.png")
    
    # --- Plot 3: Similarity Distribution by Pair Type ---
    plt.figure(figsize=(10, 6))
    df_pairs_melted = df_pairs.melt(id_vars="type", value_vars=["sim_sw_bted", "sim_sbert", "sim_tfidf"],
                                    var_name="Model", value_name="Similarity")
    model_rename = {"sim_sw_bted": "SW-BTED", "sim_sbert": "Cosine SBERT", "sim_tfidf": "Cosine TF-IDF"}
    df_pairs_melted["Model"] = df_pairs_melted["Model"].map(model_rename)
    
    sns.boxplot(data=df_pairs_melted, x="type", y="Similarity", hue="Model", palette="Set2")
    plt.title("Similarity Score Distribution (Full Text Overlap vs. Tree)", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Pair Category", fontsize=12)
    plt.ylabel("Similarity Score", fontsize=12)
    plt.ylim(-0.05, 1.05)
    
    plt.tight_layout()
    plt.savefig("results/similarity_distribution.png", dpi=300)
    plt.close()
    print("Saved similarity distribution plot to results/similarity_distribution.png")
    
    # Print formatted comparison table
    print("\n=================================== CROSS-VALIDATION REPORT ===================================")
    print(f"{'Method':<20} | {'Thresh':<6} | {'Precision':<15} | {'Recall':<15} | {'F1 Score':<15} | {'TPR A':<15} | {'TNR B':<15} | {'TNR C':<15}")
    print("-" * 130)
    for _, r in df_metrics.iterrows():
        p_str = f"{r['Precision_Mean']:.4f}±{r['Precision_Std']:.4f}" if not pd.isna(r['Precision_Std']) else f"{r['Precision_Mean']:.4f}"
        r_str = f"{r['Recall_Mean']:.4f}±{r['Recall_Std']:.4f}" if not pd.isna(r['Recall_Std']) else f"{r['Recall_Mean']:.4f}"
        f_str = f"{r['F1_Score_Mean']:.4f}±{r['F1_Score_Std']:.4f}" if not pd.isna(r['F1_Score_Std']) else f"{r['F1_Score_Mean']:.4f}"
        tpr_a = f"{r['TPR_TypeA_Mean']:.4f}±{r['TPR_TypeA_Std']:.4f}" if not pd.isna(r['TPR_TypeA_Std']) else f"{r['TPR_TypeA_Mean']:.4f}"
        tnr_b = f"{r['TNR_TypeB_Mean']:.4f}±{r['TNR_TypeB_Std']:.4f}" if not pd.isna(r['TNR_TypeB_Std']) else f"{r['TNR_TypeB_Mean']:.4f}"
        tnr_c = f"{r['TNR_TypeC_Mean']:.4f}±{r['TNR_TypeC_Std']:.4f}" if not pd.isna(r['TNR_TypeC_Std']) else f"{r['TNR_TypeC_Mean']:.4f}"
        
        print(f"{r['Method']:<20} | {r['Optimal_Threshold_Mean']:<6.2f} | {p_str:<15} | {r_str:<15} | {f_str:<15} | {tpr_a:<15} | {tnr_b:<15} | {tnr_c:<15}")
    print("===============================================================================================\n")
    
    if df_stat is not None:
        print("================================= STATISTICAL SIGNIFICANCE ===================================")
        print(f"{'Comparison':<35} | {'Chi2 Stat':<10} | {'McNemar p':<12} | {'Wilcoxon p':<12} | {'Significant':<15}")
        print("-" * 100)
        for _, r in df_stat.iterrows():
            w_p = f"{r['wilcoxon_p_value']:.4e}" if r['wilcoxon_p_value'] < 0.001 else f"{r['wilcoxon_p_value']:.4f}"
            mc_p = f"{r['p_value']:.4e}" if r['p_value'] < 0.001 else f"{r['p_value']:.4f}"
            print(f"{r['Comparison']:<35} | {r['Chi2_Statistic']:<10.4f} | {mc_p:<12} | {w_p:<12} | {r['Significant_Alpha_0.05']:<15}")
        print("===============================================================================================\n")

if __name__ == "__main__":
    evaluate()
