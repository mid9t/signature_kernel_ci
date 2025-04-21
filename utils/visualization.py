import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from typing import List, Tuple, Union, Optional

def plot_graph(graph: Union[nx.DiGraph, np.ndarray], filename: str = 'graph.png', node_names: Optional[List[str]] = None) -> plt.Figure:
    """
    Draw and save a directed causal graph:
    - Uses spring layout for node placement. :contentReference[oaicite:11]{index=11}
    - Draws directed edges with arrows. :contentReference[oaicite:12]{index=12}
    
    Args:
        graph: Either a NetworkX DiGraph or a numpy adjacency matrix
        filename: Path to save the figure
        node_names: Optional list of node names for labeling
        
    Returns:
        The matplotlib figure object
    """
    # Convert adjacency matrix to DiGraph if needed
    if isinstance(graph, np.ndarray):
        G = nx.DiGraph(graph)
        if node_names is not None:
            # Relabel nodes with names
            mapping = {i: name for i, name in enumerate(node_names)}
            G = nx.relabel_nodes(G, mapping)
    else:
        G = graph
        
    pos = nx.spring_layout(G, seed=42)  # reproducible layout :contentReference[oaicite:13]{index=13}
    plt.figure(figsize=(8, 6))
    nx.draw_networkx(
        G,
        pos=pos,
        with_labels=True,
        arrows=True,
        node_color='lightblue',
        node_size=500,
        arrowsize=20,
        font_size=12
    )                                        # draw_networkx :contentReference[oaicite:14]{index=14}
    plt.title("Causal Graph")
    plt.axis('off')  # Hide axes
    plt.tight_layout()
    plt.savefig(filename)                    # savefig :contentReference[oaicite:15]{index=15}
    fig = plt.gcf()
    return fig

def plot_hsic_distribution(
    null_stats: List[float],
    observed_stat: float,
    filename: str = 'hsic_null_dist.png'
) -> plt.Figure:
    """
    Plot histogram of bootstrap HSIC null distribution with observed statistic:
    - Histogram bins show density. :contentReference[oaicite:16]{index=16}
    - Vertical line marks observed HSIC. :contentReference[oaicite:17]{index=17}
    
    Args:
        null_stats: List of HSIC statistics from bootstrap samples
        observed_stat: The observed HSIC statistic
        filename: Path to save the figure
        
    Returns:
        The matplotlib figure object
    """
    plt.figure(figsize=(8, 6))
    plt.hist(null_stats, bins=30, density=True, alpha=0.7)  # density histogram :contentReference[oaicite:18]{index=18}
    plt.axvline(observed_stat, color='red', linestyle='--')  # observed stat line :contentReference[oaicite:19]{index=19}
    plt.title('HSIC Null Distribution')
    plt.xlabel('HSIC Statistic')
    plt.ylabel('Density')
    plt.tight_layout()
    plt.savefig(filename)                                  # save figure :contentReference[oaicite:20]{index=20}
    fig = plt.gcf()
    return fig

def plot_ci_results(
    results: List[Tuple[str, float]],
    alpha: float = 0.05,
    filename: str = 'ci_test_results.png'
) -> plt.Figure:
    """
    Plot conditional independence test results with p-values.
    
    Args:
        results: List of tuples (test_description, p_value)
        alpha: Significance level for the tests
        filename: Path to save the figure
        
    Returns:
        The matplotlib figure object
    """
    plt.figure(figsize=(10, 6))
    
    # Extract test descriptions and p-values
    test_indices = range(len(results))
    p_values = [r[1] for r in results]
    test_descriptions = [r[0] for r in results]
    
    # Create bar plot of p-values
    bars = plt.bar(test_indices, p_values, alpha=0.7)
    
    # Color bars based on significance
    for i, p in enumerate(p_values):
        if p < alpha:
            bars[i].set_color('red')  # Significant result
        else:
            bars[i].set_color('blue')  # Non-significant result
    
    # Add significance threshold line
    plt.axhline(y=alpha, color='r', linestyle='--', label=f'α = {alpha}')
    
    # Add labels and title
    plt.xlabel('Test Index')
    plt.ylabel('p-value')
    plt.title('Conditional Independence Test Results')
    plt.legend()
    
    # Add test descriptions as x-tick labels (rotated for readability)
    if len(test_descriptions) <= 10:
        plt.xticks(test_indices, test_descriptions, rotation=45, ha='right')
    else:
        plt.xticks(test_indices)
        # Add tooltip-like annotations for hover
        for i, desc in enumerate(test_descriptions):
            plt.annotate(desc, xy=(i, 0), xytext=(0, -20),
                        textcoords='offset points', ha='center', va='top',
                        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                        rotation=90)
    
    plt.tight_layout()
    plt.savefig(filename)
    fig = plt.gcf()
    return fig
