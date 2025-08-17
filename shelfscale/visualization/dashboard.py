"""
Dashboard components for interactive visualization of food data
Enhanced with upload functionality and nutrition scoring
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from dash import Dash, dcc, html, Input, Output, State, dash_table
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import base64
import io
from typing import Dict, List, Optional, Union, Any
import logging

# Lazy import heavy dependencies to avoid import errors
logger = logging.getLogger(__name__)


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


def create_nutrition_score_charts(df: pd.DataFrame) -> List[go.Figure]:
    """
    Create charts for nutrition score distributions
    
    Args:
        df: DataFrame with scoring results
        
    Returns:
        List of plotly figures
    """
    charts = []
    
    # Nutri-Score distribution
    if 'Nutri_Grade' in df.columns:
        nutri_counts = df['Nutri_Grade'].value_counts().sort_index()
        nutri_fig = px.bar(
            x=nutri_counts.index,
            y=nutri_counts.values,
            title='Nutri-Score Distribution',
            labels={'x': 'Nutri-Score Grade', 'y': 'Count'},
            color=nutri_counts.index,
            color_discrete_map={'A': 'green', 'B': 'lightgreen', 'C': 'orange', 'D': 'red', 'E': 'darkred'}
        )
        charts.append(nutri_fig)
    
    # Traffic Light summary distribution
    if 'Traffic_Lights_Summary' in df.columns:
        traffic_counts = df['Traffic_Lights_Summary'].value_counts()
        traffic_fig = px.pie(
            values=traffic_counts.values,
            names=traffic_counts.index,
            title='Traffic Light Summary Distribution',
            color=traffic_counts.index,
            color_discrete_map={'green': 'green', 'amber': 'orange', 'red': 'red'}
        )
        charts.append(traffic_fig)
    
    return charts


def create_confidence_badges(row: pd.Series) -> html.Div:
    """Create confidence badges for a product row"""
    badges = []
    
    # Weight confidence
    weight_conf = row.get('Weight_Prediction_Confidence', 0)
    weight_color = 'success' if weight_conf > 0.8 else 'warning' if weight_conf > 0.5 else 'danger'
    badges.append(
        dbc.Badge(f"Weight: {weight_conf:.1%}", color=weight_color, className="me-1")
    )
    
    # Score confidence  
    score_conf = row.get('Score_Confidence', 0)
    score_color = 'success' if score_conf > 0.8 else 'warning' if score_conf > 0.5 else 'danger'
    badges.append(
        dbc.Badge(f"Score: {score_conf:.1%}", color=score_color, className="me-1")
    )
    
    return html.Div(badges)


class ShelfScaleDashboard:
    """Enhanced dashboard for interactive visualization of ShelfScale data with upload and scoring"""
    
    def __init__(self, df: pd.DataFrame = None, summary: pd.DataFrame = None, 
                 title: str = "ShelfScale Dashboard", model_dir: str = None, output_dir: str = None):
        """
        Initialize the dashboard
        
        Args:
            df: Input DataFrame (optional for upload mode)
            summary: Summary DataFrame (optional)
            title: Dashboard title
            model_dir: Model directory for ML components
            output_dir: Output directory for exports
        """
        self.df = df if df is not None else pd.DataFrame()
        self.summary = summary if summary is not None else pd.DataFrame()
        self.title = title
        self.model_dir = model_dir
        self.output_dir = output_dir
        
        # Try to use Bootstrap theme
        try:
            self.app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        except:
            self.app = Dash(__name__)
            logger.warning("Bootstrap theme not available, using default styling")
        
        # Set up the dashboard layout
        self._setup_layout()
        
        # Set up callbacks
        self._setup_callbacks()
    
    def _setup_layout(self):
        """Set up the enhanced dashboard layout with upload and scoring"""
        # Try to get food groups, handling cases where the column might not exist
        food_groups = []
        if not self.df.empty:
            if 'Food Group' in self.df.columns:
                food_groups = sorted(self.df['Food Group'].unique())
            elif 'Food_Group' in self.df.columns:
                food_groups = sorted(self.df['Food_Group'].unique())
            elif 'Food_Category' in self.df.columns:
                food_groups = sorted(self.df['Food_Category'].unique())
        
        # Main layout with upload and data tabs
        self.app.layout = dbc.Container([
            # Title and navigation
            dbc.Row([
                dbc.Col([
                    html.H1(self.title, className="text-center mb-4"),
                    html.Hr()
                ])
            ]),
            
            # Main content tabs
            dbc.Tabs([
                # Upload tab
                dbc.Tab(label="Upload & Score", tab_id="upload", children=[
                    self._create_upload_layout()
                ]),
                
                # Visualizations tab
                dbc.Tab(label="Visualizations", tab_id="viz", children=[
                    self._create_visualization_layout(food_groups)
                ], disabled=self.df.empty),
                
                # Data table tab
                dbc.Tab(label="Data Table", tab_id="table", children=[
                    self._create_table_layout()
                ], disabled=self.df.empty),
                
                # Nutrition scores tab
                dbc.Tab(label="Nutrition Scores", tab_id="scores", children=[
                    self._create_scores_layout()
                ], disabled=self.df.empty or not self._has_scores())
            ], id="main-tabs", active_tab="upload" if self.df.empty else "viz"),
            
            # Hidden div to store uploaded data
            html.Div(id='uploaded-data', style={'display': 'none'}),
            html.Div(id='scored-data', style={'display': 'none'}),
            
        ], fluid=True)
    
    def _create_upload_layout(self):
        """Create the upload and scoring layout"""
        return dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Upload CSV/Excel File"),
                    dbc.CardBody([
                        dcc.Upload(
                            id='upload-data',
                            children=html.Div([
                                'Drag and Drop or ',
                                html.A('Select Files')
                            ]),
                            style={
                                'width': '100%',
                                'height': '60px',
                                'lineHeight': '60px',
                                'borderWidth': '1px',
                                'borderStyle': 'dashed',
                                'borderRadius': '5px',
                                'textAlign': 'center',
                                'margin': '10px'
                            },
                            multiple=False
                        ),
                        html.Div(id='upload-status'),
                        html.Hr(),
                        html.H5("Scoring Options"),
                        dbc.Row([
                            dbc.Col([
                                dbc.Checklist(
                                    options=[
                                        {"label": "Traffic Light Scores", "value": "traffic"},
                                        {"label": "Nutri-Score", "value": "nutri"},
                                    ],
                                    value=["traffic", "nutri"],
                                    id="scoring-options",
                                )
                            ], width=6),
                            dbc.Col([
                                dbc.Button("Apply Scoring", id="score-button", color="primary", 
                                          disabled=True, className="mb-2"),
                                html.Br(),
                                dbc.Button("Download Results", id="download-button", color="success", 
                                          disabled=True)
                            ], width=6)
                        ]),
                        html.Div(id='scoring-status'),
                        dcc.Download(id="download-dataframe-csv"),
                    ])
                ])
            ], width=12)
        ])
    
    def _create_visualization_layout(self, food_groups):
        """Create the visualization layout"""
        return html.Div([
            dbc.Row([
                dbc.Col([
                    html.Label('Filter by Food Group:'),
                    dcc.Dropdown(
                        id='food-group-dropdown',
                        options=[
                            {'label': group, 'value': group} for group in food_groups
                        ],
                        value=food_groups[:3] if len(food_groups) > 3 else food_groups,
                        multi=True
                    ),
                ], width=12)
            ], className="mb-3"),
            
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id='treemap-chart')
                ], width=6),
                dbc.Col([
                    dcc.Graph(id='distribution-chart')
                ], width=6)
            ])
        ])
    
    def _create_table_layout(self):
        """Create the data table layout"""
        return html.Div([
            dbc.Row([
                dbc.Col([
                    html.H4("Product Data Table"),
                    html.P("Showing processed data with weights and scores"),
                    dash_table.DataTable(
                        id='data-table-enhanced',
                        columns=[],
                        data=[],
                        sort_action="native",
                        filter_action="native",
                        page_action="native",
                        page_current=0,
                        page_size=20,
                        style_cell={'textAlign': 'left'},
                        style_data_conditional=[
                            {
                                'if': {'filter_query': '{Nutri_Grade} = A'},
                                'backgroundColor': '#d4edda',
                                'color': 'black',
                            },
                            {
                                'if': {'filter_query': '{Nutri_Grade} = E'},
                                'backgroundColor': '#f8d7da',
                                'color': 'black',
                            }
                        ]
                    )
                ])
            ])
        ])
    
    def _create_scores_layout(self):
        """Create the nutrition scores layout"""
        return html.Div([
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id='nutri-score-chart')
                ], width=6),
                dbc.Col([
                    dcc.Graph(id='traffic-light-chart')
                ], width=6)
            ]),
            dbc.Row([
                dbc.Col([
                    html.H5("Score Summary Statistics"),
                    html.Div(id='score-summary')
                ])
            ], className="mt-3")
        ])
    
    def _has_scores(self):
        """Check if the DataFrame has nutrition scores"""
        score_cols = ['Nutri_Grade', 'Traffic_Lights_Summary']
        return any(col in self.df.columns for col in score_cols)
    
    def _setup_callbacks(self):
        """Set up enhanced dashboard callbacks with upload and scoring"""
        
        # Upload file callback
        @self.app.callback(
            [Output('uploaded-data', 'children'),
             Output('upload-status', 'children'),
             Output('score-button', 'disabled')],
            Input('upload-data', 'contents'),
            State('upload-data', 'filename')
        )
        def parse_uploaded_file(contents, filename):
            if contents is None:
                return None, "", True
            
            try:
                # Parse the uploaded file
                content_type, content_string = contents.split(',')
                decoded = base64.b64decode(content_string)
                
                if filename.endswith('.csv'):
                    df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
                elif filename.endswith(('.xlsx', '.xls')):
                    df = pd.read_excel(io.BytesIO(decoded))
                else:
                    return None, dbc.Alert("Error: Please upload a CSV or Excel file", color="danger"), True
                
                # Store the data and return success message
                success_msg = dbc.Alert(
                    f"Successfully uploaded {filename} with {len(df)} rows and {len(df.columns)} columns", 
                    color="success"
                )
                
                return df.to_json(date_format='iso', orient='split'), success_msg, False
                
            except Exception as e:
                error_msg = dbc.Alert(f"Error processing file: {str(e)}", color="danger")
                return None, error_msg, True
        
        # Apply scoring callback
        @self.app.callback(
            [Output('scored-data', 'children'),
             Output('scoring-status', 'children'),
             Output('download-button', 'disabled')],
            Input('score-button', 'n_clicks'),
            [State('uploaded-data', 'children'),
             State('scoring-options', 'value')]
        )
        def apply_scoring(n_clicks, uploaded_data, scoring_options):
            if n_clicks is None or uploaded_data is None:
                raise PreventUpdate
            
            try:
                # Load the uploaded data
                df = pd.read_json(uploaded_data, orient='split')
                
                # Apply scoring
                if scoring_options:
                    score_type = 'all' if len(scoring_options) > 1 else scoring_options[0]
                    
                    # Import scoring functions here to avoid circular imports
                    from shelfscale.main import apply_nutrition_scoring
                    scored_df = apply_nutrition_scoring(df, score_type)
                    
                    success_msg = dbc.Alert(
                        f"Successfully applied {score_type} scoring to {len(scored_df)} products", 
                        color="success"
                    )
                    
                    return scored_df.to_json(date_format='iso', orient='split'), success_msg, False
                else:
                    return None, dbc.Alert("Please select scoring options", color="warning"), True
                    
            except Exception as e:
                error_msg = dbc.Alert(f"Error during scoring: {str(e)}", color="danger")
                return None, error_msg, True
        
        # Download callback
        @self.app.callback(
            Output("download-dataframe-csv", "data"),
            Input("download-button", "n_clicks"),
            State("scored-data", "children"),
            prevent_initial_call=True,
        )
        def download_scored_data(n_clicks, scored_data):
            if scored_data is None:
                raise PreventUpdate
            
            df = pd.read_json(scored_data, orient='split')
            return dcc.send_data_frame(df.to_csv, "shelfscale_scored_results.csv", index=False)
        
        # Original visualization callbacks (updated to work with stored data)
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
        self.app.run(debug=debug, port=port) 