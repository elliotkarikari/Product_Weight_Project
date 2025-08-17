#!/usr/bin/env python3
"""
Simple dashboard demonstration for ShelfScale without heavy dependencies
"""

def create_simple_dashboard():
    """Create a basic dashboard to test functionality without full dependencies"""
    
    try:
        import dash
        from dash import html, dcc, Input, Output, State, dash_table
        import plotly.express as px
        import plotly.graph_objects as go
        import pandas as pd
        import numpy as np
        import sys
        
        # Add project to path
        sys.path.insert(0, '.')
        
        # Import only the scoring functions we need
        from shelfscale.scoring import score_traffic_lights, score_nutri
        from shelfscale.data_processing.weight_extraction import WeightExtractor
        
    except ImportError as e:
        print(f"❌ Dashboard dependencies not installed: {e}")
        print("💡 Install with: pip install dash plotly pandas")
        return None
        
    # Initialize Dash app
    app = dash.Dash(__name__)
    
    # Sample data for demonstration
    sample_data = pd.DataFrame({
        'Food_Name': [
            'Apple (fresh)', 'Chocolate Bar', 'Whole Wheat Bread', 'Cola',
            'Broccoli', 'Cheddar Cheese', 'Water', 'Banana', 'Orange Juice'
        ],
        'Fat_g': [0.2, 31.0, 3.2, 0.0, 0.3, 34.4, 0.0, 0.3, 0.2],
        'SatFat_g': [0.1, 18.5, 0.7, 0.0, 0.1, 21.7, 0.0, 0.1, 0.1],
        'Sugars_g': [10.4, 43.0, 4.5, 10.6, 1.5, 0.1, 0.0, 12.2, 8.9],
        'Salt_g': [0.0, 0.02, 1.1, 0.01, 0.03, 1.8, 0.0, 0.0, 0.01],
        'Energy_kcal': [52, 534, 247, 42, 35, 416, 0, 89, 45],
        'Fiber_g': [2.4, 7.0, 8.5, 0.0, 2.8, 0.0, 0.0, 2.6, 0.2],
        'Protein_g': [0.3, 4.9, 13.0, 0.0, 2.8, 25.4, 0.0, 1.1, 0.7],
        'FVN_percent': [100, 0, 0, 0, 100, 0, 0, 100, 80]
    })
    
    # Calculate nutrition scores
    traffic_data = score_traffic_lights(sample_data)
    scored_data = score_nutri(traffic_data)
    
    # Define layout
    app.layout = html.Div([
        html.H1("🍎 ShelfScale Nutrition Dashboard", 
                style={'textAlign': 'center', 'color': '#2E8B57'}),
        
        html.P("Interactive demonstration of nutrition scoring functionality", 
               style={'textAlign': 'center', 'fontSize': '18px'}),
        
        html.Hr(),
        
        # Weight Extraction Demo
        html.Div([
            html.H2("⚖️ Weight Extraction Demo"),
            dcc.Input(
                id='weight-input',
                type='text',
                placeholder='Enter weight description (e.g., "500ml milk", "250g bread")',
                style={'width': '50%', 'marginRight': '10px'}
            ),
            html.Button('Extract Weight', id='extract-btn', n_clicks=0),
            html.Div(id='weight-output', style={'marginTop': '10px', 'fontSize': '16px'})
        ], style={'margin': '20px 0'}),
        
        html.Hr(),
        
        # Nutrition Scores Table
        html.Div([
            html.H2("📊 Nutrition Scores"),
            dash_table.DataTable(
                id='nutrition-table',
                data=scored_data.round(2).to_dict('records'),
                columns=[
                    {'name': 'Food', 'id': 'Food_Name'},
                    {'name': 'Traffic Light', 'id': 'Traffic_Lights_Summary'},
                    {'name': 'Nutri-Score', 'id': 'Nutri_Grade'},
                    {'name': 'Score Value', 'id': 'Nutri_Score'},
                    {'name': 'Fat (g)', 'id': 'Fat_g'},
                    {'name': 'Sugars (g)', 'id': 'Sugars_g'},
                    {'name': 'Salt (g)', 'id': 'Salt_g'}
                ],
                style_cell={'textAlign': 'left'},
                style_data_conditional=[
                    {
                        'if': {'filter_query': '{Traffic_Lights_Summary} = green'},
                        'backgroundColor': '#90EE90',
                        'color': 'black',
                    },
                    {
                        'if': {'filter_query': '{Traffic_Lights_Summary} = amber'},
                        'backgroundColor': '#FFD700',
                        'color': 'black',
                    },
                    {
                        'if': {'filter_query': '{Traffic_Lights_Summary} = red'},
                        'backgroundColor': '#FFB6C1',
                        'color': 'black',
                    }
                ]
            )
        ]),
        
        # Charts
        html.Div([
            html.H2("📈 Nutrition Score Distribution"),
            dcc.Graph(id='nutri-score-chart'),
            dcc.Graph(id='traffic-light-chart')
        ], style={'margin': '20px 0'}),
        
        html.Hr(),
        html.P("✨ ShelfScale: Food Product Weight Analysis & Nutrition Scoring", 
               style={'textAlign': 'center', 'color': '#666', 'marginTop': '20px'})
    ])
    
    # Weight extraction callback
    @app.callback(
        Output('weight-output', 'children'),
        Input('extract-btn', 'n_clicks'),
        State('weight-input', 'value')
    )
    def extract_weight(n_clicks, input_text):
        if n_clicks > 0 and input_text:
            try:
                extractor = WeightExtractor(target_unit='g')
                weight, unit = extractor.extract(input_text)
                
                if weight is not None:
                    return html.Div([
                        html.Span("✅ Extracted: ", style={'color': 'green'}),
                        html.Strong(f"{weight:.1f} {unit}"),
                        html.Span(f" from '{input_text}'")
                    ])
                else:
                    return html.Div([
                        html.Span("❌ No weight found in: ", style={'color': 'red'}),
                        html.Span(f"'{input_text}'")
                    ])
            except Exception as e:
                return html.Div([
                    html.Span("🛠️ Error: ", style={'color': 'orange'}),
                    html.Span(str(e))
                ])
        
        return html.Div("Enter a weight description and click Extract Weight")
    
    # Chart callbacks
    @app.callback(
        [Output('nutri-score-chart', 'figure'),
         Output('traffic-light-chart', 'figure')],
        Input('nutrition-table', 'data')
    )
    def update_charts(table_data):
        df = pd.DataFrame(table_data)
        
        # Nutri-Score distribution
        nutri_counts = df['Nutri_Grade'].value_counts()
        nutri_fig = px.bar(
            x=nutri_counts.index, 
            y=nutri_counts.values,
            title="Nutri-Score Distribution",
            labels={'x': 'Nutri-Score Grade', 'y': 'Count'},
            color=nutri_counts.index,
            color_discrete_map={'A': '#00A651', 'B': '#85BB2F', 'C': '#FCCA00', 'D': '#EE7F00', 'E': '#E63E11'}
        )
        
        # Traffic Light distribution
        traffic_counts = df['Traffic_Lights_Summary'].value_counts()
        traffic_fig = px.bar(
            x=traffic_counts.index,
            y=traffic_counts.values, 
            title="Traffic Light Distribution",
            labels={'x': 'Traffic Light Color', 'y': 'Count'},
            color=traffic_counts.index,
            color_discrete_map={'green': '#00AA00', 'amber': '#FFAA00', 'red': '#FF0000'}
        )
        
        return nutri_fig, traffic_fig
    
    return app

def main():
    """Main function to run the dashboard"""
    print("🚀 Starting ShelfScale Simple Dashboard...")
    
    app = create_simple_dashboard()
    
    if app is None:
        print("❌ Could not create dashboard due to missing dependencies")
        return
    
    print("✅ Dashboard created successfully!")
    print("🌐 Starting server on http://localhost:8050")
    print("📖 Features available:")
    print("   • Weight extraction from text")
    print("   • Nutrition scoring (Traffic Lights + Nutri-Score)")
    print("   • Interactive charts and data tables")
    print("\n💡 Press Ctrl+C to stop the server")
    
    try:
        app.run_server(debug=True, port=8050, host='0.0.0.0')
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped")
    except Exception as e:
        print(f"❌ Error running dashboard: {e}")

if __name__ == "__main__":
    main()