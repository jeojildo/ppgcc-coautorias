"""
PPGCC Coauthorship Network Analysis Pipeline
=============================================
Complete KDD pipeline for analyzing coauthorship networks among
Brazilian Computer Science graduate program faculty (2014-2023).

Usage:
    python coauthorships.py
"""

import os

import gdown
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from config import (
    DATA_DIR,
    RESULTS_DIR,
    METADATA_FILE,
    PPGCC_BRAZIL_DOWNLOAD,
    PPGCC_XLSX_DOWNLOAD,
    START_YEAR,
    END_YEAR,
    COMMUNITY_METHOD,
    COMMUNITY_RESOLUTION,
    COMMUNITY_SEED,
    COMMUNITY_COMPARISON_METHODS,
)
from coauths import (
    Selector,
    Preprocesser,
    Transformer,
    Visualizer,
)

sns.set_style("whitegrid")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# ===========================================================================
# STEP 1: SELECTION  - Download and extract Lattes XML data
# ===========================================================================

def step_selection():
    print("=" * 60)
    print("STEP 1: SELECTION")
    print("=" * 60)

    selector = Selector(
        data_dir=str(DATA_DIR),
        metadata_file=str(METADATA_FILE),
    )

    selection_dir = DATA_DIR / "01-selection"
    if selection_dir.exists() and any(selection_dir.iterdir()):
        print("  Selection data already exists, skipping download.")
        return

    print("  Downloading data from Google Drive...")
    selector.download(
        selection_dir_name="01-selection",
        gdown_url=PPGCC_BRAZIL_DOWNLOAD,
    )

    print("  Extracting ZIP files...")
    selector.extract(selection_dir_name="01-selection")

    print("  Renaming XML files...")
    selector.rename_xmls(selection_dir_name="01-selection")

    print("  Selection complete.\n")


# ===========================================================================
# STEP 2: PREPROCESSING  - Parse XMLs, build productions DataFrame
# ===========================================================================

def step_preprocessing():
    print("=" * 60)
    print("STEP 2: PREPROCESSING")
    print("=" * 60)

    preprocesser = Preprocesser(
        data_dir=str(DATA_DIR),
        metadata_file=str(METADATA_FILE),
        step_directory="02-preprocessing",
    )

    # Check if already preprocessed
    parquet_path = DATA_DIR / "02-preprocessing" / "productions.parquet"
    if parquet_path.exists():
        print("  Preprocessed data already exists, skipping.")
        return

    print("  Extracting institution registry from XMLs...")
    institution_registry = preprocesser.extract_institution_registry(
        selection_dir_name="01-selection"
    )
    print(f"  {institution_registry}\n")

    print("  Building productions DataFrame...")
    df_productions = preprocesser.frame_productions_all_institutions(institution_registry)
    print(f"  Total raw productions: {len(df_productions)}")

    df_productions = preprocesser.assign_production_ids(df_productions)
    print(f"  Unique productions: {df_productions['production_id'].nunique()}")

    df_productions = preprocesser.drop_authorless_productions(df_productions)
    print(f"  After dropping authorless: {len(df_productions)}")

    print("  Normalizing author citations...")
    df_productions = preprocesser.normalize_authors(df_productions)

    df_productions = preprocesser.explode_authors(df_productions)
    print(f"  After exploding authors: {len(df_productions)}")

    df_productions = preprocesser.filter_by_year(df_productions, START_YEAR, END_YEAR)
    print(f"  After temporal filter ({START_YEAR}-{END_YEAR}): {len(df_productions)}")

    # Save
    preprocesser.write_parquet(df_productions, step="02-preprocessing", name="productions")
    print("  Preprocessing complete.\n")


# ===========================================================================
# STEP 3: TRANSFORMATION  - Build adjacency matrix
# ===========================================================================

def step_transformation():
    print("=" * 60)
    print("STEP 3: TRANSFORMATION")
    print("=" * 60)

    transformer = Transformer(
        data_dir=str(DATA_DIR),
        metadata_file=str(METADATA_FILE),
        step_directory="03-transformation",
    )

    # Check if already transformed
    adj_path = DATA_DIR / "03-transformation" / "adjacency.parquet"
    if adj_path.exists():
        print("  Transformation data already exists, skipping.")
        return

    df_productions = transformer.read_parquet(step="02-preprocessing", name="productions")

    print("  Building entity IDs...")
    df_productions = transformer.build_entity_ids(df_productions)
    print(f"  Unique researchers: {df_productions['nid'].nunique()}")

    flag_productions = transformer.build_flag_productions(df_productions)

    transformer.write_parquet(df_productions, step="03-transformation", name="productions")

    # Build adjacency matrix (with deduplication)
    print("  Building adjacency matrix...")
    df_adjacency = transformer.build_adjacency(df_productions, flag_productions, remove_isolated=True)
    print(f"  Adjacency matrix shape: {df_adjacency.shape}")

    max_col = df_adjacency.max().idxmax()
    max_row = df_adjacency[max_col].idxmax()
    print(f"  Max coauthorship: nodes ({max_row}, {max_col}) = {df_adjacency[max_col][max_row]}")

    transformer.write_parquet(df_adjacency, step="03-transformation", name="adjacency")
    print("  Transformation complete.\n")


# ===========================================================================
# STEP 4: VISUALIZATION - Network analysis
# ===========================================================================

def step_visualization():
    print("=" * 60)
    print("STEP 4: VISUALIZATION")
    print("=" * 60)

    vis_dir = RESULTS_DIR / "network"
    os.makedirs(vis_dir, exist_ok=True)

    visualizer = Visualizer(
        data_dir=str(DATA_DIR),
        metadata_file=str(METADATA_FILE),
        step_directory="04-visualization",
    )
    visualizer.figure_directory = vis_dir

    df_productions = visualizer.read_parquet(step="03-transformation", name="productions")
    df_adjacency = visualizer.read_parquet(step="03-transformation", name="adjacency")

    # --- Degree distribution ---
    print("  Computing degree distribution...")
    visualizer.plot_degree_distribution(df_adjacency)

    # --- Shortest paths ---
    print("  Computing all-pairs shortest path lengths...")
    df_shortest_paths = visualizer.compute_shortest_paths(df_adjacency)
    print(f"  Shortest path pairs: {len(df_shortest_paths)}")
    print(df_shortest_paths["length"].value_counts().sort_index())
    visualizer.plot_shortest_path_histogram(df_shortest_paths)
    visualizer.plot_shortest_path_violin(df_shortest_paths)

    # --- Density ---
    density = visualizer.compute_density(df_adjacency)
    print(f"  Network density: {density:.6f} ({density*100:.4f}%)")

    # --- Degree histogram ---
    print("  Plotting degree histogram...")
    visualizer.plot_degree_histogram(df_adjacency)

    # --- Betweenness centrality ---
    print("  Computing betweenness centrality...")
    df_betweenness, corr = visualizer.plot_betweenness_centrality(df_adjacency, df_productions)
    print(f"  Pearson r={corr['pearson_r']:.3f} (p={corr['pearson_p']:.2e})")
    print(f"  Spearman rho={corr['spearman_r']:.3f} (p={corr['spearman_p']:.2e})")

    # --- Overall coauthorship network graph ---
    print("  Plotting overall coauthorship network...")
    visualizer.plot_coauthorship(df_adjacency, figsize=(10, 10), filename="coauthorship_network_overall")
    plt.close("all")

    print("  Visualization complete.\n")


# ===========================================================================
# STEP 5: COMMUNITY DETECTION
# ===========================================================================

def step_community_detection():
    print("=" * 60)
    print("STEP 5: COMMUNITY DETECTION")
    print("=" * 60)

    comm_dir = RESULTS_DIR / "communities"
    os.makedirs(comm_dir, exist_ok=True)

    visualizer = Visualizer(
        data_dir=str(DATA_DIR),
        metadata_file=str(METADATA_FILE),
        step_directory="04-visualization",
    )
    visualizer.figure_directory = comm_dir

    df_productions = visualizer.read_parquet(step="03-transformation", name="productions")
    df_adjacency = visualizer.read_parquet(step="03-transformation", name="adjacency")

    # Run comparison
    print("  Comparing community detection methods...")
    method_params = {
        m: {"resolution": COMMUNITY_RESOLUTION, "seed": COMMUNITY_SEED}
        for m in COMMUNITY_COMPARISON_METHODS
    }
    df_comparison = visualizer.compare_community_methods(
        df_adjacency,
        methods=COMMUNITY_COMPARISON_METHODS,
        method_params=method_params,
    )
    print(df_comparison.to_string(index=False))
    df_comparison.to_csv(comm_dir / "community_comparison.csv", index=False)

    # Best method by modularity
    best_method = df_comparison.iloc[0]["method"]
    print(f"\n  Best method by modularity: {best_method}")

    # --- Full community analysis ---
    print("  Running full community analysis...")
    results = visualizer.run_full_community_analysis(
        df_adjacency, df_productions, method=best_method,
        resolution=COMMUNITY_RESOLUTION, seed=COMMUNITY_SEED,
    )

    report = results["report"]
    print(f"  Communities: {int(report['num_communities'])}")
    print(f"  Modularity: {report['modularity']:.4f}")
    print(f"  Coverage: {report['coverage']:.4f}")
    print(f"  Performance: {report['performance']:.4f}")

    # Save community assignments
    results["df_communities"].to_csv(comm_dir / "community_assignments.csv", index=False)
    pd.DataFrame([report]).to_csv(comm_dir / "community_report.csv", index=False)

    # Table 1: Community structural profile
    df_structural = results["df_structural"]
    df_structural.to_csv(comm_dir / "community_structural_profile.csv", index=False)
    print("\n  Table 1 – Community structural profile:")
    print(df_structural.head(10).to_string(index=False))

    # Table 2: Institutional profile per community
    df_inst_profile = results["df_inst_profile"]
    df_inst_profile.to_csv(comm_dir / "community_institutional_profile.csv", index=False)
    print("\n  Table 2 – Institutional profile per community:")
    print(df_inst_profile.head(10).to_string(index=False))

    # Table 3: Institution network metrics
    df_inst_metrics = results["df_inst_metrics"]
    df_inst_metrics.to_csv(comm_dir / "institution_network_metrics.csv", index=False)
    print("\n  Table 3 – Top institutions by collaboration volume:")
    print(df_inst_metrics.head(20).to_string(index=False))

    # Node roles
    df_roles = results["df_roles"]
    df_roles.to_csv(comm_dir / "node_roles.csv", index=False)
    print("\n  Node role distribution:")
    print(df_roles["role"].value_counts().to_string())

    # Scatter plot: P vs z_intra
    print("\n  Generating roles scatter plot...")
    visualizer.plot_roles_scatter(df_roles, filename="fig_roles_scatter")

    # Network visualization colored by community
    print("  Generating community network visualization...")
    df_edges = pd.DataFrame(
        [(u, v, d.get("weight", 1)) for u, v, d in nx.from_pandas_adjacency(df_adjacency).edges(data=True)],
        columns=["source", "target", "n_coauthorships"],
    )
    visualizer.plot_coauthorship_network(
        df_edges,
        community_method=best_method,
        community_kwargs={"resolution": COMMUNITY_RESOLUTION, "seed": COMMUNITY_SEED},
        filename=f"coauthorship_network_{best_method}",
    )
    plt.close("all")

    print("  Community detection complete.\n")


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    print("\nPPGCC Coauthorship Network Analysis Pipeline")
    print("=" * 60)
    print(f"  Data directory:    {DATA_DIR}")
    print(f"  Results directory: {RESULTS_DIR}")
    print(f"  Period:            {START_YEAR}-{END_YEAR}")
    print()

    # Download ppgcc.xlsx from Google Drive if not present
    ppgcc_dst = DATA_DIR / "ppgcc.xlsx"
    if not ppgcc_dst.exists():
        print("  Downloading ppgcc.xlsx from Google Drive...")
        gdown.download(PPGCC_XLSX_DOWNLOAD, str(ppgcc_dst), quiet=False)
        print(f"  Saved to {ppgcc_dst}")

    step_selection()
    step_preprocessing()
    step_transformation()
    step_visualization()
    step_community_detection()

    print("=" * 60)
    print("PIPELINE COMPLETE")
    print(f"  Data artifacts:   {DATA_DIR}")
    print(f"  Result artifacts: {RESULTS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
