from typing import Dict, Optional
import plotly.graph_objects as go
from core.solution_representation import VRPSolution
from roots import get_project_root
from utils.metrics import calculate_fitness


def get_color_palette(n: int, color_type: str = 'red') -> list:
    if color_type == 'red':
        base_colors = ['#ffcccb', '#ff9999', '#ff6666', '#ff3333', '#ff0000',
                       '#cc0000', '#990000', '#660000', '#330000']
    else:
        base_colors = ['#90ee90', '#7ccd7c', '#66cc66', '#4dbd4d', '#33aa33',
                       '#2d9e2d', '#248f24', '#1a7f1a', '#116611']

    if n <= len(base_colors):
        return base_colors[:n]

    indices = np.linspace(0, len(base_colors) - 1, n).astype(int)
    return [base_colors[i] for i in indices]


def plot_optimal(instance: Dict, save: bool = False) -> go.Figure:
    if 'optimal_routes' not in instance or instance['optimal_routes'] is None:
        raise ValueError("No optimal solution found in instance data")

    coords = instance['coordinates']
    optimal_routes = instance['optimal_routes']

    optimal_solution = VRPSolution(
        routes=optimal_routes,
        instance_data=instance
    )

    fitness = calculate_fitness(optimal_solution, instance)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=[coords[0][0]],
        y=[coords[0][1]],
        mode='markers+text',
        marker=dict(size=20, color='black', symbol='circle-open', line=dict(width=3)),
        text=['Depot'],
        textposition='top center',
        name='Depot',
        showlegend=True
    ))

    colors = get_color_palette(len(optimal_routes), 'green')

    for route_idx, route in enumerate(optimal_routes):
        route_coords = [coords[0]] + [coords[c] for c in route] + [coords[0]]
        xs = [c[0] for c in route_coords]
        ys = [c[1] for c in route_coords]

        fig.add_trace(go.Scatter(
            x=xs,
            y=ys,
            mode='lines',
            line=dict(color=colors[route_idx], width=2, dash='dash'),
            name=f'Route {route_idx + 1}',
            showlegend=False
        ))

    customer_coords = coords[1:]
    customer_ids = list(range(1, len(coords)))

    fig.add_trace(go.Scatter(
        x=[c[0] for c in customer_coords],
        y=[c[1] for c in customer_coords],
        mode='markers+text',
        marker=dict(size=10, color='lightblue', line=dict(width=1, color='black')),
        text=[str(i) for i in customer_ids],
        textposition='middle center',
        textfont=dict(size=8),
        name='Customers',
        showlegend=True
    ))

    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='lines',
        line=dict(color='green', width=2, dash='dash'),
        name=f'Optimal: Fitness={fitness:.2f}, Routes={len(optimal_routes)}',
        showlegend=True
    ))

    fig.update_layout(
        title=f"Optimal Solution - {instance.get('name', 'Unknown')}",
        xaxis_title="X Coordinate",
        yaxis_title="Y Coordinate",
        hovermode='closest',
        showlegend=True,
        width=900,
        height=700
    )

    if save:
        _save_figure(fig, instance.get('name', 'unknown'), 'optimal')

    return fig


def plot_solution(instance: Dict,
                  solution: VRPSolution,
                  plot_against_optimal: bool = True,
                  save: bool = True) -> go.Figure:
    coords = instance['coordinates']
    solution_routes = solution.routes
    solution_fitness = calculate_fitness(solution, instance)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=[coords[0][0]],
        y=[coords[0][1]],
        mode='markers+text',
        marker=dict(size=20, color='black', symbol='circle-open', line=dict(width=3)),
        text=['Depot'],
        textposition='top center',
        name='Depot',
        showlegend=True
    ))

    if plot_against_optimal:
        if 'optimal_routes' not in instance or instance['optimal_routes'] is None:
            raise ValueError("No optimal solution found for comparison")

        optimal_routes = instance['optimal_routes']
        optimal_solution = VRPSolution(routes=optimal_routes, instance_data=instance)
        optimal_fitness = calculate_fitness(optimal_solution, instance)

        green_colors = get_color_palette(len(optimal_routes), 'green')

        for route_idx, route in enumerate(optimal_routes):
            route_coords = [coords[0]] + [coords[c] for c in route] + [coords[0]]
            xs = [c[0] for c in route_coords]
            ys = [c[1] for c in route_coords]

            fig.add_trace(go.Scatter(
                x=xs,
                y=ys,
                mode='lines',
                line=dict(color=green_colors[route_idx], width=2, dash='dash'),
                name=f'Optimal Route {route_idx + 1}',
                showlegend=False
            ))

        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='lines',
            line=dict(color='green', width=2, dash='dash'),
            name=f'Optimal: Fitness={optimal_fitness:.2f}, Routes={len(optimal_routes)}',
            showlegend=True
        ))

    red_colors = get_color_palette(len(solution_routes), 'red')

    for route_idx, route in enumerate(solution_routes):
        route_coords = [coords[0]] + [coords[c] for c in route] + [coords[0]]
        xs = [c[0] for c in route_coords]
        ys = [c[1] for c in route_coords]

        fig.add_trace(go.Scatter(
            x=xs,
            y=ys,
            mode='lines',
            line=dict(color=red_colors[route_idx], width=2.5),
            name=f'Solution Route {route_idx + 1}',
            showlegend=False
        ))

    customer_coords = coords[1:]
    customer_ids = list(range(1, len(coords)))

    fig.add_trace(go.Scatter(
        x=[c[0] for c in customer_coords],
        y=[c[1] for c in customer_coords],
        mode='markers+text',
        marker=dict(size=10, color='lightblue', line=dict(width=1, color='black')),
        text=[str(i) for i in customer_ids],
        textposition='middle center',
        textfont=dict(size=8),
        name='Customers',
        showlegend=True
    ))

    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='lines',
        line=dict(color='red', width=2.5),
        name=f'Found: Fitness={solution_fitness:.2f}, Routes={len(solution_routes)}',
        showlegend=True
    ))

    title_suffix = " vs Optimal" if plot_against_optimal else ""
    fig.update_layout(
        title=f"Solution{title_suffix} - {instance.get('name', 'Unknown')}",
        xaxis_title="X Coordinate",
        yaxis_title="Y Coordinate",
        hovermode='closest',
        showlegend=True,
        width=900,
        height=700
    )

    if save:
        suffix = 'comparison' if plot_against_optimal else 'solution'
        _save_figure(fig, instance.get('name', 'unknown'), suffix)

    return fig


def _save_figure(fig: go.Figure, instance_name: str, suffix: str) -> None:
    root = get_project_root()
    plot_dir = root / "results" / "plots" / instance_name
    plot_dir.mkdir(parents=True, exist_ok=True)

    html_path = plot_dir / f"{suffix}.html"
    png_path = plot_dir / f"{suffix}.png"

    fig.write_html(str(html_path))
    fig.write_image(str(png_path))

    print(f"Saved plots to:")
    print(f"  HTML: {html_path}")
    print(f"  PNG: {png_path}")


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import List, Union


def plot_num_feasible(num_feasible: int, num_infeasible: int) -> None:
    categories: List[str] = ['Feasible', 'Infeasible']
    counts: List[int] = [num_feasible, num_infeasible]
    colors: List[str] = ['#2ecc71', '#e74c3c']

    plt.figure(figsize=(8, 6))
    bars = plt.bar(categories, counts, color=colors)
    plt.ylabel('Number of Solutions')
    plt.title('Solution Feasibility Distribution')

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{int(height)}', ha='center', va='bottom')

    plt.tight_layout()
    plt.show()


def plot_boxplot(data: Union[np.ndarray, pd.Series, List[float]], title: Optional[str]) -> None:
    plt.figure(figsize=(6, 8))
    sns.boxplot(y=data, color='#ab9f52')
    plt.ylabel('Value')
    if title is not None:
        plt.title(title)
    else:
        plt.title('Distribution Analysis')
    plt.tight_layout()
    plt.show()


def plot_boxplots(data_list: List[Union[np.ndarray, pd.Series, List[float]]], title: Optional[str]) -> None:
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=data_list, palette='Set2')
    plt.ylabel('Value')
    plt.xlabel('Experiment Group')
    if title is None:
        plt.title('Comparative Distribution Analysis')
    else:
        plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_pairs(exp1: Union[np.ndarray, pd.Series, List[float]],
               exp2: Union[np.ndarray, pd.Series, List[float]]) -> None:
    if len(exp1) != len(exp2):
        raise ValueError("Experiment lists must have the same length for pairing.")

    n: int = len(exp1)
    index: np.ndarray = np.arange(n)
    bar_width: float = 0.35

    plt.figure(figsize=(12, 6))
    plt.bar(index, exp1, bar_width, label='Exp 1 (Vanilla)', color='#95a5a6')
    plt.bar(index + bar_width, exp2, bar_width, label='Exp 2 (RL-Tuned)', color='#3498db')

    plt.xlabel('Instance Index')
    plt.ylabel('Fitness / Cost')
    plt.title('Paired Performance Comparison')
    names = ['Medians', 'Means']
    plt.xticks(index + bar_width / 2, [f'{names[i]}' for i in range(n)])
    plt.legend()

    plt.tight_layout()
    plt.show()


def plot_bars(list_of_vals: List[float], list_of_col_names: List[str], main_title: str) -> None:
    if len(list_of_vals) != len(list_of_col_names):
        raise ValueError("Values and column names must have the same length.")

    plt.figure(figsize=(10, 6), facecolor='white')
    sns.barplot(x=list_of_col_names, y=list_of_vals, palette='viridis')

    plt.title(main_title)
    plt.xticks(rotation=45)
    plt.ylabel('Value')

    plt.tight_layout()
    plt.show()