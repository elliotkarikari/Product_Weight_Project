"""
Visualizations for model performance tracking and metrics
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Union, Tuple, Any
import pickle
import json
import datetime
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from shelfscale.utils.learning import get_performance_trend, load_performance_history

class PerformanceTracker:
    """Track and visualize model performance metrics over time"""
    
    def __init__(self, 
                metrics_path: str = "output/performance_metrics.json",
                feature_history_path: str = "output/feature_history.pkl"):
        """
        Initialize the performance tracker
        
        Args:
            metrics_path: Path to save metrics history
            feature_history_path: Path to save feature importance history
        """
        self.metrics_path = metrics_path
        self.feature_history_path = feature_history_path
        self.metrics_history = self._load_metrics()
        self.feature_history = self._load_features()
        
        # Create output directories if they don't exist
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
        
    def _load_metrics(self) -> List[Dict[str, Any]]:
        """Load metrics history from file"""
        try:
            if os.path.exists(self.metrics_path):
                with open(self.metrics_path, 'r') as f:
                    return json.load(f)
            return []
        except Exception as e:
            print(f"Error loading metrics history: {e}")
            return []
    
    def _load_features(self) -> Dict[str, List[Tuple[datetime.datetime, Dict[str, float]]]]:
        """Load feature importance history from file"""
        try:
            if os.path.exists(self.feature_history_path):
                with open(self.feature_history_path, 'rb') as f:
                    return pickle.load(f)
            return {}
        except Exception as e:
            print(f"Error loading feature history: {e}")
            return {}
    
    def _save_metrics(self) -> None:
        """Save metrics history to file"""
        try:
            with open(self.metrics_path, 'w') as f:
                json.dump(self.metrics_history, f, indent=2)
        except Exception as e:
            print(f"Error saving metrics history: {e}")
    
    def _save_features(self) -> None:
        """Save feature importance history to file"""
        try:
            with open(self.feature_history_path, 'wb') as f:
                pickle.dump(self.feature_history, f)
        except Exception as e:
            print(f"Error saving feature history: {e}")
    
    def add_metrics(self, metrics: Dict[str, float], source: str = "training") -> None:
        """
        Add new metrics to history
        
        Args:
            metrics: Dictionary of metric values
            source: Source of metrics (e.g., "training", "evaluation")
        """
        metrics_entry = {
            "date": datetime.datetime.now().isoformat(),
            "source": source,
            "metrics": metrics
        }
        
        self.metrics_history.append(metrics_entry)
        self._save_metrics()
    
    def add_feature_importance(self, feature_importance: Dict[str, float]) -> None:
        """
        Add new feature importance to history
        
        Args:
            feature_importance: Dictionary of feature importance values
        """
        now = datetime.datetime.now()
        
        # Initialize if empty
        if not self.feature_history:
            self.feature_history = {"dates": [], "data": {}}
        
        # Add date
        self.feature_history["dates"].append(now)
        
        # Add importance for each feature
        for feature, importance in feature_importance.items():
            if feature not in self.feature_history["data"]:
                self.feature_history["data"][feature] = []
            self.feature_history["data"][feature].append(importance)
        
        self._save_features()
    
    def plot_metrics_over_time(self, 
                             metrics: List[str] = None, 
                             figsize: Tuple[int, int] = (12, 6),
                             output_file: str = None) -> None:
        """
        Plot metrics over time
        
        Args:
            metrics: List of metrics to plot (if None, plot all)
            figsize: Figure size
            output_file: Path to save figure (if None, display only)
        """
        if not self.metrics_history:
            print("No metrics history available")
            return
        
        # Extract dates and metric values
        dates = [datetime.datetime.fromisoformat(entry["date"]) for entry in self.metrics_history]
        
        # Get all available metrics if none specified
        if metrics is None:
            metrics = set()
            for entry in self.metrics_history:
                metrics.update(entry["metrics"].keys())
            metrics = list(metrics)
        
        # Set up plot
        plt.figure(figsize=figsize)
        
        # Plot each metric
        for metric in metrics:
            values = [entry["metrics"].get(metric, None) for entry in self.metrics_history]
            
            # Filter out None values
            valid_points = [(d, v) for d, v in zip(dates, values) if v is not None]
            if valid_points:
                plot_dates, plot_values = zip(*valid_points)
                plt.plot(plot_dates, plot_values, marker='o', label=metric)
        
        plt.title("Performance Metrics Over Time")
        plt.xlabel("Date")
        plt.ylabel("Value")
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        if output_file:
            plt.savefig(output_file)
        else:
            plt.show()
        plt.close()
    
    def plot_feature_importance(self,
                              top_n: int = 10,
                              figsize: Tuple[int, int] = (12, 8),
                              output_file: str = None) -> None:
        """
        Plot feature importance
        
        Args:
            top_n: Number of top features to plot
            figsize: Figure size
            output_file: Path to save figure (if None, display only)
        """
        if not self.feature_history or not self.feature_history["data"]:
            print("No feature importance history available")
            return
        
        # Calculate average importance for each feature
        avg_importance = {}
        for feature, values in self.feature_history["data"].items():
            if values:
                avg_importance[feature] = sum(values) / len(values)
        
        # Sort features by importance
        sorted_features = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)
        
        # Get top N features
        top_features = sorted_features[:top_n]
        
        # Set up plot
        plt.figure(figsize=figsize)
        
        # Plot feature importance
        features, importances = zip(*top_features)
        plt.barh(features, importances)
        plt.title(f"Top {top_n} Feature Importance")
        plt.xlabel("Importance")
        plt.grid(True, axis='x')
        
        plt.tight_layout()
        if output_file:
            plt.savefig(output_file)
        else:
            plt.show()
        plt.close()
    
    def plot_feature_importance_over_time(self,
                                        top_n: int = 5,
                                        figsize: Tuple[int, int] = (12, 8),
                                        output_file: str = None) -> None:
        """
        Plot feature importance over time
        
        Args:
            top_n: Number of top features to plot
            figsize: Figure size
            output_file: Path to save figure (if None, display only)
        """
        if not self.feature_history or not self.feature_history["data"]:
            print("No feature importance history available")
            return
        
        # Calculate average importance for each feature
        avg_importance = {}
        for feature, values in self.feature_history["data"].items():
            if values:
                avg_importance[feature] = sum(values) / len(values)
        
        # Sort features by importance
        sorted_features = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)
        
        # Get top N features
        top_features = [feature for feature, _ in sorted_features[:top_n]]
        
        # Set up plot
        plt.figure(figsize=figsize)
        
        # Plot each feature's importance over time
        for feature in top_features:
            if feature in self.feature_history["data"]:
                plt.plot(self.feature_history["dates"], self.feature_history["data"][feature], marker='o', label=feature)
        
        plt.title(f"Top {top_n} Feature Importance Over Time")
        plt.xlabel("Date")
        plt.ylabel("Importance")
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        if output_file:
            plt.savefig(output_file)
        else:
            plt.show()
        plt.close()
    
    def create_interactive_dashboard(self, output_file: str = "output/performance_dashboard.html") -> None:
        """
        Create an interactive Plotly dashboard
        
        Args:
            output_file: Path to save HTML dashboard
        """
        if not self.metrics_history:
            print("No metrics history available")
            return
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Metrics Over Time", "Feature Importance", 
                           "Training Samples", "Confusion Matrix"),
            specs=[[{"type": "scatter"}, {"type": "bar"}],
                  [{"type": "scatter"}, {"type": "heatmap"}]]
        )
        
        # 1. Metrics over time
        dates = [datetime.datetime.fromisoformat(entry["date"])
                 for entry in self.metrics_history]
        metrics = set()
        for entry in self.metrics_history:
            metrics.update(entry["metrics"].keys())
        
        for metric in metrics:
            values = [entry["metrics"].get(metric, None) for entry in self.metrics_history]
            # Filter out None values
            valid_points = [(d, v) for d, v in zip(dates, values) if v is not None]
            if valid_points:
                plot_dates, plot_values = zip(*valid_points)
                fig.add_trace(
                    go.Scatter(x=plot_dates, y=plot_values, mode='lines+markers', name=metric),
                    row=1, col=1
                )
        
        # 2. Feature importance
        if self.feature_history and self.feature_history["data"]:
            avg_importance = {}
            for feature, values in self.feature_history["data"].items():
                if values:
                    avg_importance[feature] = sum(values) / len(values)
            
            # Sort features by importance
            sorted_features = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)
            
            # Get top 10 features
            top_features = sorted_features[:10]
            features, importances = zip(*top_features)
            
            fig.add_trace(
                go.Bar(y=features, x=importances, orientation='h'),
                row=1, col=2
            )
        
        # 3. Training samples over time
        sample_counts = [entry["metrics"].get("training_samples", None) for entry in self.metrics_history]
        valid_samples = [(d, s) for d, s in zip(dates, sample_counts) if s is not None]
        if valid_samples:
            sample_dates, samples = zip(*valid_samples)
            fig.add_trace(
                go.Scatter(x=sample_dates, y=samples, mode='lines+markers', name='Training Samples'),
                row=2, col=1
            )
        
        # 4. Last confusion matrix (if available)
        if self.metrics_history and "confusion_matrix" in self.metrics_history[-1]["metrics"]:
            cm = self.metrics_history[-1]["metrics"]["confusion_matrix"]
            try:
                cm_array = np.asarray(cm)
                if cm_array.ndim != 2:
                    raise ValueError("Confusion matrix must be 2-D")
                
                # Generate labels based on matrix shape
                n_classes = cm_array.shape[0]
                pred_labels = [f"Predicted {i}" for i in range(n_classes)]
                actual_labels = [f"Actual {i}" for i in range(n_classes)]
                
                fig.add_trace(
                    go.Heatmap(z=cm_array, x=pred_labels, 
                              y=actual_labels, colorscale="Blues"),
                    row=2, col=2
                )
            except Exception as e:
                print(f"Error creating confusion matrix plot: {e}")
        
        fig.update_layout(height=800, width=1200, title_text="Model Performance Dashboard")
        fig.write_html(output_file)
        print(f"Interactive dashboard saved to {output_file}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the latest performance metrics
        
        Returns:
            Summary of latest metrics
        """
        if not self.metrics_history:
            return {"error": "No metrics history available"}
        
        latest_metrics = self.metrics_history[-1]["metrics"]
        
        # Calculate improvement over time if we have at least 2 entries
        improvements = {}
        if len(self.metrics_history) >= 2:
            first_metrics = self.metrics_history[0]["metrics"]
            for metric, value in latest_metrics.items():
                if metric in first_metrics and isinstance(value, (int, float)):
                    improvements[metric] = value - first_metrics[metric]
        
        return {
            "latest_metrics": latest_metrics,
            "date": self.metrics_history[-1]["date"],
            "improvements": improvements,
            "total_runs": len(self.metrics_history)
        }

def plot_performance_trend(output_path: str = "output/performance_trend.png") -> None:
    """
    Plot the performance trend over time
    
    Args:
        output_path: Path to save the plot
    """
    # Get performance trend data
    trend = get_performance_trend()
    
    # Check if we have data
    if not trend["timestamps"]:
        print("No performance history available.")
        return
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Plot metrics
    plt.subplot(2, 1, 1)
    plt.plot(trend["timestamps"], trend["accuracy"], label="Accuracy", marker="o")
    plt.plot(trend["timestamps"], trend["precision"], label="Precision", marker="s")
    plt.plot(trend["timestamps"], trend["recall"], label="Recall", marker="^")
    plt.plot(trend["timestamps"], trend["f1"], label="F1 Score", marker="d", linewidth=2)
    
    plt.title("Model Performance Metrics Over Time")
    plt.xlabel("Timestamp")
    plt.ylabel("Score")
    plt.ylim(0, 1.05)
    plt.legend(loc="lower right")
    plt.grid(True, linestyle="--", alpha=0.7)
    
    # Rotate x-axis labels for readability
    plt.xticks(rotation=45)
    
    # Plot match count
    plt.subplot(2, 1, 2)
    plt.bar(trend["timestamps"], trend["match_count"], color="skyblue")
    plt.title("Number of Training Matches")
    plt.xlabel("Timestamp")
    plt.ylabel("Count")
    plt.grid(True, axis="y", linestyle="--", alpha=0.7)
    
    # Rotate x-axis labels for readability
    plt.xticks(rotation=45)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    print(f"Performance trend plot saved to {output_path}")
    plt.close()

def plot_feature_importance(output_path: str = "output/feature_importance.png") -> None:
    """
    Plot feature importance from the latest model
    
    Args:
        output_path: Path to save the plot
    """
    # Get performance history
    history = load_performance_history()
    
    if not history:
        print("No performance history available.")
        return
    
    # Get the latest record
    latest = history[-1]
    
    # Check if we have feature importance data
    if "feature_importance" not in latest:
        # Try to load from the model file directly
        from shelfscale.matching.algorithm import FoodMatcher
        matcher = FoodMatcher()
        
        if not matcher.feature_importance:
            print("No feature importance data available.")
            return
        
        feature_importance = matcher.feature_importance
    else:
        feature_importance = latest["feature_importance"]
    
    # Create dataframe for plotting
    features_df = pd.DataFrame({
        "Feature": list(feature_importance.keys()),
        "Importance": list(feature_importance.values())
    })
    
    # Sort by importance
    features_df = features_df.sort_values("Importance", ascending=False)
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Plot horizontal bars
    plt.barh(features_df["Feature"], features_df["Importance"], color="skyblue")
    plt.title("Feature Importance")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.grid(True, axis="x", linestyle="--", alpha=0.7)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    print(f"Feature importance plot saved to {output_path}")
    plt.close()

def generate_performance_report(output_path: str = "output/performance_report.html") -> None:
    """
    Generate an HTML performance report
    
    Args:
        output_path: Path to save the HTML report
    """
    # Get performance history
    history = load_performance_history()
    
    if not history:
        print("No performance history available.")
        return
    
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame()
    
    for record in history:
        entry = {
            "timestamp": record["timestamp"],
            **record["metrics"],
            "sources": ", ".join(record["dataset_info"]["sources"]),
            "total_counts": sum(record["dataset_info"]["source_counts"].values())
        }
        df = pd.concat([df, pd.DataFrame([entry])], ignore_index=True)
    
    # Calculate improvement
    if len(df) > 1:
        df["f1_change"] = df["f1"].diff()
        df["accuracy_change"] = df["accuracy"].diff()
    
    # Generate HTML
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Model Performance Report</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            h1 { color: #333; }
            .summary { background-color: #f5f5f5; padding: 15px; border-radius: 5px; }
            table { border-collapse: collapse; width: 100%; margin-top: 20px; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #f2f2f2; }
            tr:nth-child(even) { background-color: #f9f9f9; }
            .positive { color: green; }
            .negative { color: red; }
            .charts { margin-top: 30px; }
            .chart { margin-bottom: 20px; }
        </style>
    </head>
    <body>
        <h1>ShelfScale Model Performance Report</h1>
        
        <div class="summary">
    """
    
    # Add summary statistics
    if not df.empty:
        latest = df.iloc[-1]
        first = df.iloc[0]
        
        # Calculate total improvement
        total_f1_improvement = latest["f1"] - first["f1"] if "f1" in latest and "f1" in first else 0
        
        html += f"""
            <h2>Summary</h2>
            <p><strong>Latest Performance:</strong> F1 Score: {latest.get('f1', 0):.3f}, Accuracy: {latest.get('accuracy', 0):.3f}</p>
            <p><strong>Total Improvement:</strong> {total_f1_improvement:.3f} F1 Score improvement since first recording</p>
            <p><strong>Latest Training:</strong> {latest.get('total_counts', 0)} total matches across {latest.get('sources', 'N/A')} sources</p>
        """
    
    html += """
        </div>
        
        <h2>Performance History</h2>
        <table>
            <tr>
                <th>Timestamp</th>
                <th>F1 Score</th>
                <th>Change</th>
                <th>Accuracy</th>
                <th>Precision</th>
                <th>Recall</th>
                <th>Matches</th>
                <th>Sources</th>
            </tr>
    """
    
    # Add table rows
    for _, row in df.iterrows():
        f1_change = row.get("f1_change", 0)
        change_class = "positive" if f1_change > 0 else "negative" if f1_change < 0 else ""
        change_symbol = "▲" if f1_change > 0 else "▼" if f1_change < 0 else "-"
        
        html += f"""
            <tr>
                <td>{row.get('timestamp', '')}</td>
                <td>{row.get('f1', 0):.3f}</td>
                <td class="{change_class}">{change_symbol} {abs(f1_change):.3f if f1_change else '-'}</td>
                <td>{row.get('accuracy', 0):.3f}</td>
                <td>{row.get('precision', 0):.3f}</td>
                <td>{row.get('recall', 0):.3f}</td>
                <td>{row.get('total_counts', 0)}</td>
                <td>{row.get('sources', 'N/A')}</td>
            </tr>
        """
    
    html += """
        </table>
        
        <div class="charts">
            <h2>Performance Visualization</h2>
            <div class="chart">
                <img src="performance_trend.png" alt="Performance Trend" style="max-width: 100%;">
            </div>
            <div class="chart">
                <img src="feature_importance.png" alt="Feature Importance" style="max-width: 100%;">
            </div>
        </div>
    </body>
    </html>
    """
    
    # Save the HTML file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(html)
    
    print(f"Performance report saved to {output_path}")

def plot_match_quality_distribution(match_df: pd.DataFrame, 
                                   similarity_col: str = 'Similarity_Score',
                                   output_file: str = None) -> None:
    """
    Plot distribution of match quality scores
    
    Args:
        match_df: DataFrame with match results
        similarity_col: Column name for similarity scores
        output_file: Path to save figure (if None, display only)
    """
    plt.figure(figsize=(10, 6))
    
    # Plot histogram
    sns.histplot(match_df[similarity_col], bins=20, kde=True)
    
    plt.title("Distribution of Match Quality Scores")
    plt.xlabel("Similarity Score")
    plt.ylabel("Frequency")
    plt.grid(True)
    
    if output_file:
        plt.savefig(output_file)
    else:
        plt.show()
    plt.close()

def plot_weight_prediction_accuracy(predicted: np.ndarray, 
                                   actual: np.ndarray,
                                   output_file: str = None) -> None:
    """
    Plot weight prediction accuracy
    
    Args:
        predicted: Array of predicted weights
        actual: Array of actual weights
        output_file: Path to save figure (if None, display only)
    """
    plt.figure(figsize=(10, 8))
    
    # Scatter plot of predicted vs actual
    plt.scatter(actual, predicted, alpha=0.5)
    
    # Perfect prediction line
    max_val = max(np.max(predicted), np.max(actual))
    plt.plot([0, max_val], [0, max_val], 'r--', label='Perfect Prediction')
    
    plt.title("Weight Prediction Accuracy")
    plt.xlabel("Actual Weight (g)")
    plt.ylabel("Predicted Weight (g)")
    plt.grid(True)
    plt.legend()
    
    # Calculate metrics
    mse = np.mean((predicted - actual) ** 2)
    mae = np.mean(np.abs(predicted - actual))
    r2 = 1 - (np.sum((actual - predicted) ** 2) / np.sum((actual - np.mean(actual)) ** 2))
    
    plt.figtext(0.15, 0.85, f"MSE: {mse:.2f}\nMAE: {mae:.2f}\nR²: {r2:.3f}", 
               bbox=dict(facecolor='white', alpha=0.8))
    
    if output_file:
        plt.savefig(output_file)
    else:
        plt.show()
    plt.close()

def create_matching_confusion_matrix(matches_df: pd.DataFrame,
                                    predicted_col: str = 'predicted_match',
                                    actual_col: str = 'correct_match',
                                    output_file: str = None) -> None:
    """
    Create confusion matrix for matching results
    
    Args:
        matches_df: DataFrame with matching results
        predicted_col: Column name for predicted matches
        actual_col: Column name for actual matches
        output_file: Path to save figure (if None, display only)
    """
    if predicted_col not in matches_df.columns or actual_col not in matches_df.columns:
        print(f"Error: required columns {predicted_col} or {actual_col} not found")
        return
    
    # Create confusion matrix
    cm = pd.crosstab(matches_df[actual_col], matches_df[predicted_col], 
                    rownames=['Actual'], colnames=['Predicted'])
    
    # Plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    
    plt.title('Matching Confusion Matrix')
    
    if output_file:
        plt.savefig(output_file)
    else:
        plt.show()
    plt.close()
    
    # Calculate and return metrics
    tp = cm.iloc[1, 1] if cm.shape == (2, 2) else 0
    fp = cm.iloc[0, 1] if cm.shape == (2, 2) else 0
    fn = cm.iloc[1, 0] if cm.shape == (2, 2) else 0
    tn = cm.iloc[0, 0] if cm.shape == (2, 2) else 0
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "confusion_matrix": cm.values.tolist(),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    }

if __name__ == "__main__":
    # If run directly, generate all visualizations
    plot_performance_trend()
    plot_feature_importance()
    generate_performance_report() 