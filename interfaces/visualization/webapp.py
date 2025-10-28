"""
Dash Web Application for Knowledge Graph Visualization

Interactive web interface for 3D knowledge graph visualization with real-time updates.
"""

import dash
from dash import dcc, html, Input, Output, State
import logging
from .graph_3d import KnowledgeGraph3D


class GraphVisualizationApp:
    """
    Dash application for interactive knowledge graph visualization.
    
    Features:
    - Real-time graph updates
    - 3D interactive visualization
    - Parameter controls
    - Statistics display
    """
    
    def __init__(self, title: str = "A-LMI Knowledge Graph"):
        """
        Initialize Dash app.
        
        Args:
            title: Application title
        """
        self.logger = logging.getLogger(__name__)
        self.title = title
        
        # Initialize Dash app
        self.app = dash.Dash(__name__, title=title)
        
        # Knowledge graph visualizer
        self.knowledge_graph = KnowledgeGraph3D()
        
        # Setup layout
        self._setup_layout()
        
        # Setup callbacks
        self._setup_callbacks()
        
        self.logger.info("Dash web app initialized")
    
    def _setup_layout(self):
        """Setup Dash app layout."""
        self.app.layout = html.Div([
            # Header
            html.H1(
                self.title,
                style={'textAlign': 'center', 'margin': '20px'}
            ),
            
            # Controls
            html.Div([
                html.Button('Refresh Graph', id='refresh-btn', n_clicks=0),
                html.Button('Export Graph', id='export-btn', n_clicks=0),
                dcc.Interval(
                    id='interval-component',
                    interval=30*1000,  # Update every 30 seconds
                    n_intervals=0
                )
            ], style={'margin': '20px', 'textAlign': 'center'}),
            
            # Graph visualization
            html.Div([
                dcc.Graph(id='knowledge-graph-3d')
            ]),
            
            # Statistics
            html.Div([
                html.H3('Statistics'),
                html.Div(id='stats-display')
            ], style={'margin': '20px'})
        ])
    
    def _setup_callbacks(self):
        """Setup Dash callbacks for interactivity."""
        @self.app.callback(
            Output('knowledge-graph-3d', 'figure'),
            Input('refresh-btn', 'n_clicks'),
            Input('interval-component', 'n_intervals')
        )
        def update_graph(n_clicks, n_intervals):
            """Update graph visualization."""
            return self.knowledge_graph.create_3d_graph()
        
        @self.app.callback(
            Output('stats-display', 'children'),
            Input('interval-component', 'n_intervals')
        )
        def update_stats(n_intervals):
            """Update statistics display."""
            num_nodes = len(self.knowledge_graph.nodes)
            num_edges = len(self.knowledge_graph.edges)
            
            return html.Div([
                html.P(f"Nodes: {num_nodes}"),
                html.P(f"Edges: {num_edges}"),
                html.P(f"Last updated: {dash.callback_context.triggered[0]['prop_id'] if dash.callback_context.triggered else 'Never'}")
            ])
    
    def run(self, host: str = '127.0.0.1', port: int = 8050, debug: bool = True):
        """
        Run the Dash application.
        
        Args:
            host: Host to bind to
            port: Port to listen on
            debug: Enable debug mode
        """
        self.logger.info(f"Starting Dash app on http://{host}:{port}")
        self.app.run_server(host=host, port=port, debug=debug)


def main():
    """Run the visualization web app."""
    app = GraphVisualizationApp()
    app.run()


if __name__ == "__main__":
    main()

