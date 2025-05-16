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
    
    def __init__(self, df: pd.DataFrame, summary: pd.DataFrame = None, title: str = "ShelfScale Dashboard"):
        """
        Initialize the dashboard
        
        Args:
            df: Input DataFrame
            summary: Summary DataFrame (optional)
            title: Dashboard title
        """
        self.df = df
        self.summary = summary if summary is not None else pd.DataFrame()
        self.title = title
        self.app = Dash(__name__)
        
        # Set up the dashboard layout
        self._setup_layout()
        
        # Set up callbacks
        self._setup_callbacks()
    
    def _setup_layout(self):
        """Set up the dashboard layout"""
        # Try to get food groups, handling cases where the column might not exist
        food_groups = []
        if 'Food Group' in self.df.columns:
            food_groups = sorted(self.df['Food Group'].unique())
        elif 'Food_Group' in self.df.columns:
            food_groups = sorted(self.df['Food_Group'].unique())
        elif 'Food_Category' in self.df.columns:
            food_groups = sorted(self.df['Food_Category'].unique())
        
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
            # Identify the correct group column
            group_col = self._get_group_column()
            weight_col = self._get_weight_column()
            
            if not group_col or not weight_col:
                # Return empty figure if necessary columns aren't found
                return go.Figure().update_layout(title="No suitable data columns found")
            
            # Filter data if groups are selected
            filtered_df = self.df
            if selected_groups:
                if isinstance(selected_groups, list):
                    filtered_df = self.df[self.df[group_col].isin(selected_groups)]
                else:
                    filtered_df = self.df[self.df[group_col] == selected_groups]
            
            # Create the treemap
            try:
                fig = create_food_group_treemap(filtered_df, group_col, weight_col)
                # Prevent memory leaks in matplotlib backend
                import matplotlib.pyplot as plt
                plt.close('all')
                return fig
            except Exception as e:
                # Return empty figure with error message
                return go.Figure().update_layout(title=f"Error creating treemap: {str(e)}")
        
        @self.app.callback(
            Output('distribution-chart', 'figure'),
            Input('food-group-dropdown', 'value')
        )
        def update_distribution(selected_groups):
            # Identify the correct group column
            group_col = self._get_group_column()
            weight_col = self._get_weight_column()
            
            if not group_col or not weight_col:
                # Return empty figure if necessary columns aren't found
                return go.Figure().update_layout(title="No suitable data columns found")
            
            # Filter data if groups are selected
            filtered_df = self.df
            if selected_groups:
                if isinstance(selected_groups, list):
                    filtered_df = self.df[self.df[group_col].isin(selected_groups)]
                else:
                    filtered_df = self.df[self.df[group_col] == selected_groups]
            
            # Create the distribution chart
            try:
                fig = create_weight_distribution_chart(filtered_df, group_col, weight_col)
                # Prevent memory leaks in matplotlib backend
                import matplotlib.pyplot as plt
                plt.close('all')
                return fig
            except Exception as e:
                # Return empty figure with error message
                return go.Figure().update_layout(title=f"Error creating distribution chart: {str(e)}")
        
        @self.app.callback(
            Output('data-table', 'children'),
            Input('food-group-dropdown', 'value')
        )
        def update_table(selected_groups):
            # Identify the correct group column
            group_col = self._get_group_column()
            
            # Filter data if groups are selected
            filtered_df = self.df
            if selected_groups and group_col:
                if isinstance(selected_groups, list):
                    filtered_df = self.df[self.df[group_col].isin(selected_groups)]
                else:
                    filtered_df = self.df[self.df[group_col] == selected_groups]
            
            # Limit the displayed columns to make the table readable
            display_cols = self._get_display_columns(filtered_df)
            display_df = filtered_df[display_cols]
            
            # Create a table from the DataFrame
            try:
                rows = []
                # Header
                rows.append(html.Tr([html.Th(col) for col in display_cols]))
                
                # Body - limit to 100 rows for performance
                for i in range(min(100, len(display_df))):
                    rows.append(html.Tr([html.Td(str(display_df.iloc[i][col])) for col in display_cols]))
                
                return html.Table(rows, style={'width': '100%', 'border': '1px solid black'})
            except Exception as e:
                return html.Div(f"Error creating table: {str(e)}")
    
    def _get_group_column(self):
        """Find the appropriate food group column in the DataFrame"""
        possible_cols = ['Food Group', 'Food_Group', 'Food_Category', 'Super_Category']
        for col in possible_cols:
            if col in self.df.columns:
                return col
        return None
    
    def _get_weight_column(self):
        """Find the appropriate weight column in the DataFrame"""
        possible_cols = ['Normalized_Weight', 'Weight_Value', 'Weight_g', 'Weight']
        for col in possible_cols:
            if col in self.df.columns:
                return col
        return None
    
    def _get_display_columns(self, df, max_cols=10):
        """Get a subset of columns for display in the data table"""
        # Priority columns to always include if available
        priority_cols = ['Food Name', 'Food_Name', 'Food Group', 'Food_Group', 'Food_Category', 
                       'Weight_Value', 'Weight_g', 'Normalized_Weight']
        
        # Filter to only include columns that exist in the DataFrame
        available_priority = [col for col in priority_cols if col in df.columns]
        
        # If we have too few priority columns, add some other columns
        if len(available_priority) < max_cols:
            remaining_cols = [col for col in df.columns if col not in available_priority]
            # Add remaining columns up to the max_cols limit
            available_priority.extend(remaining_cols[:max_cols - len(available_priority)])
        
        # If we have too many columns, truncate to max_cols
        return available_priority[:max_cols]
    
    def run_server(self, debug: bool = True, port: int = 8050):
        """
        Run the dashboard server
        
        Args:
            debug: Enable debug mode
            port: Server port
        """
        self.app.run_server(debug=debug, port=port) 