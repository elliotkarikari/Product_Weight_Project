"""
Enhanced ShelfScale Dashboard - Interactive Food Product Weight Analysis
Improved version based on Jupyter notebook design with better UX and visual design
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from dash import Dash, dcc, html, Input, Output, State, dash_table, callback_context
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import base64
import io
import os
from typing import Dict, List, Optional, Union, Any
import logging
# import matplotlib.cm as cm  # Not needed for this implementation

logger = logging.getLogger(__name__)

class EnhancedShelfScaleDashboard:
    """Enhanced dashboard with pre-built product database and improved UX"""
    
    def __init__(self, data_file: str = None):
        """
        Initialize the enhanced dashboard
        
        Args:
            data_file: Path to the data file (defaults to dp_full.csv)
        """
        # Set default data file path
        if data_file is None:
            data_file = "/mnt/c/Users/ellio/Desktop/Freelance/Product_Weight_Project/Data/Processed/ReducedwithWeights/dp_full.csv"
        
        self.data_file = data_file
        self.df = self._load_data()
        
        # Initialize Dash app with modern theme
        external_stylesheets = [
            dbc.themes.BOOTSTRAP,
            dbc.icons.FONT_AWESOME,
            "https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap"
        ]
        
        self.app = Dash(__name__, external_stylesheets=external_stylesheets)
        self.app.title = "ShelfScale - Food Product Weight Analysis"
        
        # Add custom CSS
        self._add_custom_css()
        
        # Set up layout and callbacks
        self._setup_layout()
        self._setup_callbacks()
    
    def _load_data(self) -> pd.DataFrame:
        """Load the product weight dataset"""
        try:
            if os.path.exists(self.data_file):
                df = pd.read_csv(self.data_file, index_col=0)
                logger.info(f"Loaded {len(df)} products from {self.data_file}")
                return df
            else:
                logger.warning(f"Data file {self.data_file} not found. Using empty dataset.")
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return pd.DataFrame()
    
    def _add_custom_css(self):
        """Add custom CSS styling"""
        self.app.index_string = '''
        <!DOCTYPE html>
        <html>
            <head>
                {%metas%}
                <title>{%title%}</title>
                {%favicon%}
                {%css%}
                <style>
                    * {
                        font-family: 'Inter', sans-serif;
                    }
                    
                    .main-header {
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        color: white;
                        padding: 2rem 0;
                        margin-bottom: 2rem;
                        border-radius: 15px;
                        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
                    }
                    
                    .main-title {
                        font-size: 2.5rem;
                        font-weight: 700;
                        text-align: center;
                        margin: 0;
                    }
                    
                    .main-subtitle {
                        font-size: 1.2rem;
                        font-weight: 300;
                        text-align: center;
                        margin-top: 0.5rem;
                        opacity: 0.9;
                    }
                    
                    .stats-card {
                        background: white;
                        border-radius: 15px;
                        padding: 1.5rem;
                        box-shadow: 0 5px 15px rgba(0,0,0,0.08);
                        border: none;
                        transition: transform 0.3s ease;
                    }
                    
                    .stats-card:hover {
                        transform: translateY(-5px);
                    }
                    
                    .stat-number {
                        font-size: 2.5rem;
                        font-weight: 700;
                        color: #667eea;
                        margin: 0;
                    }
                    
                    .stat-label {
                        font-size: 0.9rem;
                        font-weight: 500;
                        color: #6c757d;
                        text-transform: uppercase;
                        letter-spacing: 0.5px;
                    }
                    
                    .treemap-container {
                        background: white;
                        border-radius: 15px;
                        padding: 1.5rem;
                        box-shadow: 0 5px 15px rgba(0,0,0,0.08);
                        margin-bottom: 2rem;
                    }
                    
                    .controls-panel {
                        background: #f8f9fa;
                        border-radius: 15px;
                        padding: 1.5rem;
                        margin-bottom: 2rem;
                    }
                    
                    .search-box {
                        border-radius: 10px;
                        border: 2px solid #e9ecef;
                        padding: 0.75rem 1rem;
                        font-size: 1rem;
                        transition: border-color 0.3s ease;
                    }
                    
                    .search-box:focus {
                        border-color: #667eea;
                        box-shadow: 0 0 0 0.2rem rgba(102, 126, 234, 0.25);
                    }
                    
                    .btn-modern {
                        border-radius: 10px;
                        padding: 0.75rem 1.5rem;
                        font-weight: 500;
                        text-transform: uppercase;
                        letter-spacing: 0.5px;
                        transition: all 0.3s ease;
                    }
                    
                    .btn-primary-modern {
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        border: none;
                        color: white;
                    }
                    
                    .btn-primary-modern:hover {
                        transform: translateY(-2px);
                        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
                    }
                    
                    .modal-modern .modal-content {
                        border-radius: 20px;
                        border: none;
                        box-shadow: 0 20px 60px rgba(0,0,0,0.15);
                    }
                    
                    .modal-modern .modal-header {
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        color: white;
                        border-radius: 20px 20px 0 0;
                        border: none;
                    }
                    
                    .data-table-container {
                        border-radius: 15px;
                        overflow: hidden;
                        box-shadow: 0 5px 15px rgba(0,0,0,0.08);
                    }
                    
                    .loading-spinner {
                        display: flex;
                        justify-content: center;
                        align-items: center;
                        height: 200px;
                    }
                </style>
            </head>
            <body>
                {%app_entry%}
                <footer>
                    {%config%}
                    {%scripts%}
                    {%renderer%}
                </footer>
            </body>
        </html>
        '''
    
    def _create_treemap(self) -> go.Figure:
        """Create an interactive treemap of food groups"""
        if self.df.empty:
            return go.Figure()
        
        # Count products by food group
        food_group_counts = self.df['Food Group'].value_counts()
        
        # Create labels and sizes
        labels = food_group_counts.index
        sizes = food_group_counts.values
        
        # Generate beautiful colors using a color palette
        colors = px.colors.qualitative.Set3[:len(labels)]
        
        # Create the treemap
        fig = go.Figure(go.Treemap(
            labels=labels,
            parents=[''] * len(labels),
            values=sizes,
            texttemplate="%{label}<br>%{value} products",
            textfont=dict(size=14, family="Inter"),
            marker=dict(
                colors=colors,
                line=dict(width=2, color='white'),
                colorscale='Viridis'
            ),
            hovertemplate="<b>%{label}</b><br>Products: %{value}<br><extra></extra>",
        ))
        
        # Update layout for modern look
        fig.update_layout(
            margin=dict(t=20, l=10, r=10, b=10),
            font=dict(family="Inter", size=12),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        
        return fig
    
    def _create_stats_cards(self) -> html.Div:
        """Create statistics cards showing dataset overview"""
        if self.df.empty:
            return html.Div("No data available", className="text-center text-muted")
        
        total_products = len(self.df)
        total_groups = self.df['Food Group'].nunique()
        products_with_weights = len(self.df[self.df['Product Weight'].notna()])
        coverage_percent = round((products_with_weights / total_products) * 100, 1)
        
        return dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H2(f"{total_products:,}", className="stat-number"),
                        html.P("Total Products", className="stat-label mb-0")
                    ])
                ], className="stats-card text-center")
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H2(f"{total_groups}", className="stat-number"),
                        html.P("Food Groups", className="stat-label mb-0")
                    ])
                ], className="stats-card text-center")
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H2(f"{products_with_weights:,}", className="stat-number"),
                        html.P("With Weight Data", className="stat-label mb-0")
                    ])
                ], className="stats-card text-center")
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H2(f"{coverage_percent}%", className="stat-number"),
                        html.P("Data Coverage", className="stat-label mb-0")
                    ])
                ], className="stats-card text-center")
            ], width=3),
        ], className="mb-4")
    
    def _setup_layout(self):
        """Set up the enhanced dashboard layout"""
        # Get unique food groups for dropdown
        food_groups = sorted(self.df['Food Group'].unique()) if not self.df.empty else []
        
        self.app.layout = dbc.Container([
            # Header
            html.Div([
                html.H1("ShelfScale", className="main-title"),
                html.P("Comprehensive Food Product Weight Analysis Dashboard", 
                      className="main-subtitle")
            ], className="main-header"),
            
            # Statistics Cards
            self._create_stats_cards(),
            
            # Controls Panel
            dbc.Card([
                dbc.CardBody([
                    html.H5("Dashboard Controls", className="mb-3"),
                    dbc.Row([
                        dbc.Col([
                            html.Label("Search Products:", className="form-label"),
                            dbc.Input(
                                id="product-search",
                                placeholder="Search by product name...",
                                type="text",
                                className="search-box"
                            )
                        ], width=4),
                        dbc.Col([
                            html.Label("Filter by Food Group:", className="form-label"),
                            dcc.Dropdown(
                                id="food-group-filter",
                                options=[{"label": "All Groups", "value": "all"}] + 
                                        [{"label": group, "value": group} for group in food_groups],
                                value="all",
                                clearable=False
                            )
                        ], width=4),
                        dbc.Col([
                            html.Label("Actions:", className="form-label"),
                            html.Br(),
                            dbc.Button(
                                "Download Full Dataset",
                                id="download-full-btn",
                                color="primary",
                                className="btn-modern btn-primary-modern me-2"
                            ),
                            dcc.Download(id="download-full-data")
                        ], width=4)
                    ])
                ])
            ], className="controls-panel"),
            
            # Main Treemap
            dbc.Card([
                dbc.CardBody([
                    html.H5("Food Groups Overview", className="mb-3"),
                    html.P("Click on any food group to explore products in detail", 
                          className="text-muted mb-3"),
                    dcc.Graph(
                        id="food-groups-treemap",
                        figure=self._create_treemap(),
                        config={'displayModeBar': False}
                    )
                ])
            ], className="treemap-container"),
            
            # Product Details Modal
            dbc.Modal([
                dbc.ModalHeader([
                    html.H4("Food Group Details", id="modal-title")
                ], className="modal-header"),
                dbc.ModalBody([
                    html.Div(id="modal-content")
                ]),
                dbc.ModalFooter([
                    dbc.Button(
                        "Download Group Data",
                        id="download-group-btn",
                        color="primary",
                        className="btn-modern btn-primary-modern me-2"
                    ),
                    dbc.Button(
                        "Close",
                        id="close-modal-btn",
                        color="secondary",
                        className="btn-modern"
                    )
                ])
            ], id="product-modal", size="xl", className="modal-modern"),
            
            # Hidden div to store current group data
            html.Div(id="current-group-data", style={"display": "none"}),
            dcc.Download(id="download-group-data")
            
        ], fluid=True, className="px-4")
    
    def _setup_callbacks(self):
        """Set up dashboard callbacks"""
        
        @self.app.callback(
            [Output('product-modal', 'is_open'),
             Output('modal-title', 'children'),
             Output('modal-content', 'children'),
             Output('current-group-data', 'children')],
            [Input('food-groups-treemap', 'clickData'),
             Input('close-modal-btn', 'n_clicks')],
            [State('product-modal', 'is_open')]
        )
        def toggle_modal(click_data, close_clicks, is_open):
            """Handle treemap clicks and modal opening/closing"""
            ctx = callback_context
            
            if not ctx.triggered:
                raise PreventUpdate
            
            trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
            
            if trigger_id == 'close-modal-btn':
                return False, "", "", ""
            
            if trigger_id == 'food-groups-treemap' and click_data:
                # Extract the clicked food group
                food_group = click_data['points'][0]['label']
                
                # Filter data for this group
                group_data = self.df[self.df['Food Group'] == food_group].copy()
                
                # Create a preview table (first 15 rows)
                preview_data = group_data.head(15)
                
                # Create the table
                table = dash_table.DataTable(
                    data=preview_data.to_dict('records'),
                    columns=[
                        {"name": "Food Code", "id": "Food Code"},
                        {"name": "Food Name", "id": "Food Name"},
                        {"name": "Product Weight", "id": "Product Weight"},
                        {"name": "Source", "id": "Source"},
                        {"name": "Pack Size", "id": "Pack Size"}
                    ],
                    style_table={'overflowX': 'auto'},
                    style_cell={
                        'textAlign': 'left',
                        'padding': '12px',
                        'fontFamily': 'Inter'
                    },
                    style_header={
                        'backgroundColor': '#667eea',
                        'color': 'white',
                        'fontWeight': 'bold'
                    },
                    style_data_conditional=[
                        {
                            'if': {'row_index': 'odd'},
                            'backgroundColor': '#f8f9fa'
                        }
                    ],
                    page_size=10
                )
                
                # Create summary info
                total_products = len(group_data)
                with_weights = len(group_data[group_data['Product Weight'].notna()])
                
                summary = dbc.Alert([
                    html.H6(f"📊 Summary for {food_group}", className="alert-heading"),
                    html.P(f"Total Products: {total_products:,}"),
                    html.P(f"Products with Weight Data: {with_weights:,}"),
                    html.P(f"Coverage: {(with_weights/total_products*100):.1f}%")
                ], color="info", className="mb-3")
                
                modal_content = [
                    summary,
                    html.P(f"Showing first 15 products out of {total_products:,} total:", 
                          className="text-muted mb-3"),
                    table
                ]
                
                return True, f"{food_group} - Product Details", modal_content, group_data.to_json()
            
            return is_open, "", "", ""
        
        @self.app.callback(
            Output('download-full-data', 'data'),
            Input('download-full-btn', 'n_clicks'),
            prevent_initial_call=True
        )
        def download_full_dataset(n_clicks):
            """Download the full dataset"""
            if n_clicks:
                return dcc.send_data_frame(
                    self.df.to_csv, 
                    "shelfscale_full_dataset.csv",
                    index=False
                )
        
        @self.app.callback(
            Output('download-group-data', 'data'),
            Input('download-group-btn', 'n_clicks'),
            State('current-group-data', 'children'),
            State('modal-title', 'children'),
            prevent_initial_call=True
        )
        def download_group_dataset(n_clicks, group_data_json, modal_title):
            """Download the current food group dataset"""
            if n_clicks and group_data_json:
                group_data = pd.read_json(group_data_json)
                food_group = modal_title.split(' - ')[0] if ' - ' in modal_title else 'food_group'
                filename = f"shelfscale_{food_group.replace(' ', '_').lower()}_data.csv"
                
                return dcc.send_data_frame(
                    group_data.to_csv,
                    filename,
                    index=False
                )
        
        @self.app.callback(
            Output('food-groups-treemap', 'figure'),
            [Input('product-search', 'value'),
             Input('food-group-filter', 'value')]
        )
        def update_treemap(search_term, food_group_filter):
            """Update treemap based on filters"""
            filtered_df = self.df.copy()
            
            # Apply search filter
            if search_term:
                filtered_df = filtered_df[
                    filtered_df['Food Name'].str.contains(search_term, case=False, na=False)
                ]
            
            # Apply food group filter
            if food_group_filter and food_group_filter != "all":
                filtered_df = filtered_df[filtered_df['Food Group'] == food_group_filter]
            
            # Create updated treemap
            if filtered_df.empty:
                return go.Figure().add_annotation(
                    text="No products found matching your criteria",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, xanchor='center', yanchor='middle',
                    showarrow=False, font=dict(size=16)
                )
            
            # Count products by food group
            food_group_counts = filtered_df['Food Group'].value_counts()
            
            # Create labels and sizes
            labels = food_group_counts.index
            sizes = food_group_counts.values
            
            # Generate colors
            colors = px.colors.qualitative.Set3[:len(labels)]
            
            # Create the treemap
            fig = go.Figure(go.Treemap(
                labels=labels,
                parents=[''] * len(labels),
                values=sizes,
                texttemplate="%{label}<br>%{value} products",
                textfont=dict(size=14, family="Inter"),
                marker=dict(
                    colors=colors,
                    line=dict(width=2, color='white')
                ),
                hovertemplate="<b>%{label}</b><br>Products: %{value}<br><extra></extra>",
            ))
            
            # Update layout
            fig.update_layout(
                margin=dict(t=20, l=10, r=10, b=10),
                font=dict(family="Inter", size=12),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            
            return fig
    
    def run_server(self, debug: bool = True, port: int = 8050):
        """Run the dashboard server"""
        self.app.run(debug=debug, port=port)


def create_enhanced_dashboard(data_file: str = None) -> EnhancedShelfScaleDashboard:
    """Create and return an enhanced dashboard instance"""
    return EnhancedShelfScaleDashboard(data_file)


if __name__ == "__main__":
    # Create and run the enhanced dashboard
    dashboard = create_enhanced_dashboard()
    dashboard.run_server(debug=True, port=8050)