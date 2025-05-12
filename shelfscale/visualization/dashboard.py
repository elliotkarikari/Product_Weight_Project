"""
Dashboard components for interactive visualization of food data
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from dash import Dash, dcc, html, Input, Output
from typing import Dict, List, Optional, Union, Any


def create_food_group_treemap(df: pd.DataFrame, 
                             group_col: str = 'Food Group',
                             weight_col: str = 'Normalized_Weight') -> go.Figure:
    """
    Create a treemap visualization of food groups by weight
    
    Args:
        df: Input DataFrame
        group_col: Name of the food group column
        weight_col: Name of the weight column
        
    Returns:
        Plotly figure with treemap
    """
    if group_col not in df.columns:
        raise ValueError(f"Column '{group_col}' not found in DataFrame")
        
    if weight_col not in df.columns:
        raise ValueError(f"Column '{weight_col}' not found in DataFrame")
    
    # Group by food group and sum weights
    group_weights = df.groupby(group_col)[weight_col].sum().reset_index()
    
    # Create treemap
    fig = px.treemap(
        group_weights,
        path=[group_col],
        values=weight_col,
        color=weight_col,
        color_continuous_scale='RdBu',
        title='Food Groups by Weight'
    )
    
    # Update layout
    fig.update_layout(
        margin=dict(t=50, l=25, r=25, b=25),
        font=dict(size=14)
    )
    
    return fig


def create_weight_distribution_chart(df: pd.DataFrame, 
                                    group_col: str = 'Food Group', 
                                    weight_col: str = 'Normalized_Weight',
                                    top_n: int = 10) -> go.Figure:
    """
    Create a chart showing the weight distribution across food groups
    
    Args:
        df: Input DataFrame
        group_col: Name of the food group column
        weight_col: Name of the weight column
        top_n: Number of top groups to show
        
    Returns:
        Plotly figure with distribution chart
    """
    if group_col not in df.columns:
        raise ValueError(f"Column '{group_col}' not found in DataFrame")
        
    if weight_col not in df.columns:
        raise ValueError(f"Column '{weight_col}' not found in DataFrame")
    
    # Group by food group and calculate statistics
    stats = df.groupby(group_col)[weight_col].agg(['mean', 'median', 'std', 'count']).reset_index()
    
    # Sort by count (frequency) and select top N
    top_groups = stats.sort_values('count', ascending=False).head(top_n)
    
    # Create bar chart
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=top_groups[group_col],
        y=top_groups['mean'],
        name='Mean',
        error_y=dict(type='data', array=top_groups['std']),
        marker_color='rgb(55, 83, 109)'
    ))
    
    fig.add_trace(go.Bar(
        x=top_groups[group_col],
        y=top_groups['median'],
        name='Median',
        marker_color='rgb(26, 118, 255)'
    ))
    
    # Update layout
    fig.update_layout(
        title='Weight Distribution by Food Group',
        xaxis=dict(
            title='Food Group',
            tickfont=dict(size=14),
            tickangle=-45
        ),
        yaxis=dict(title='Weight'),
        legend=dict(
            x=0.01,
            y=0.99,
            bgcolor='rgba(255, 255, 255, 0.5)',
            bordercolor='rgba(0, 0, 0, 0.1)'
        ),
        barmode='group'
    )
    
    return fig


class ShelfScaleDashboard:
    """Dashboard for interactive visualization of ShelfScale data"""
    
    def __init__(self, df: pd.DataFrame, title: str = "ShelfScale Dashboard"):
        """
        Initialize the dashboard
        
        Args:
            df: Input DataFrame
            title: Dashboard title
        """
        self.df = df
        self.title = title
        self.app = Dash(__name__)
        
        # Set up the dashboard layout
        self._setup_layout()
        
        # Set up callbacks
        self._setup_callbacks()
    
    def _setup_layout(self):
        """Set up the dashboard layout"""
        # Get list of food groups for dropdown
        food_groups = sorted(self.df['Food Group'].unique())
        
        self.app.layout = html.Div([
            # Title
            html.H1(self.title, style={'textAlign': 'center'}),
            
            # Dropdown for selecting food group
            html.Div([
                html.Label('Select Food Group:'),
                dcc.Dropdown(
                    id='food-group-dropdown',
                    options=[
                        {'label': group, 'value': group} for group in food_groups
                    ],
                    value=food_groups[0] if food_groups else None,
                    multi=True
                ),
            ], style={'padding': '10px', 'width': '50%', 'margin': 'auto'}),
            
            # Tabs for different visualizations
            dcc.Tabs([
                dcc.Tab(label='Treemap', children=[
                    dcc.Graph(id='treemap-chart')
                ]),
                dcc.Tab(label='Distribution', children=[
                    dcc.Graph(id='distribution-chart')
                ]),
                dcc.Tab(label='Data Table', children=[
                    html.Div(id='data-table')
                ])
            ])
        ])
    
    def _setup_callbacks(self):
        """Set up dashboard callbacks"""
        @self.app.callback(
            Output('treemap-chart', 'figure'),
            Input('food-group-dropdown', 'value')
        )
        def update_treemap(selected_groups):
            # Filter data if groups are selected
            filtered_df = self.df
            if selected_groups:
                if isinstance(selected_groups, list):
                    filtered_df = self.df[self.df['Food Group'].isin(selected_groups)]
                else:
                    filtered_df = self.df[self.df['Food Group'] == selected_groups]
            
            # Create the treemap
            return create_food_group_treemap(filtered_df)
        
        @self.app.callback(
            Output('distribution-chart', 'figure'),
            Input('food-group-dropdown', 'value')
        )
        def update_distribution(selected_groups):
            # Filter data if groups are selected
            filtered_df = self.df
            if selected_groups:
                if isinstance(selected_groups, list):
                    filtered_df = self.df[self.df['Food Group'].isin(selected_groups)]
                else:
                    filtered_df = self.df[self.df['Food Group'] == selected_groups]
            
            # Create the distribution chart
            return create_weight_distribution_chart(filtered_df)
        
        @self.app.callback(
            Output('data-table', 'children'),
            Input('food-group-dropdown', 'value')
        )
        def update_table(selected_groups):
            # Filter data if groups are selected
            filtered_df = self.df
            if selected_groups:
                if isinstance(selected_groups, list):
                    filtered_df = self.df[self.df['Food Group'].isin(selected_groups)]
                else:
                    filtered_df = self.df[self.df['Food Group'] == selected_groups]
            
            # Create a table from the DataFrame
            table = html.Table(
                # Header
                [html.Tr([html.Th(col) for col in filtered_df.columns])] +
                
                # Body
                [html.Tr([
                    html.Td(filtered_df.iloc[i][col]) for col in filtered_df.columns
                ]) for i in range(min(100, len(filtered_df)))]
            )
            
            return table
    
    def run_server(self, debug: bool = True, port: int = 8050):
        """
        Run the dashboard server
        
        Args:
            debug: Enable debug mode
            port: Server port
        """
        self.app.run_server(debug=debug, port=port) 