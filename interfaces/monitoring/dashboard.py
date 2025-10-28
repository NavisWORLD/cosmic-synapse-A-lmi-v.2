"""
Monitoring Dashboard

Web-based dashboard for real-time system monitoring.
"""

import logging
from dash import Dash, html, dcc
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import json
from typing import Dict, Any

from .metrics import MetricsCollector


class MonitoringDashboard:
    """
    Web dashboard for monitoring A-LMI system.
    
    Displays:
    - Kafka message rates
    - Processing throughput
    - Memory usage
    - Error rates
    - Hypothesis generation stats
    """
    
    def __init__(self, collector: MetricsCollector, port: int = 8051):
        """
        Initialize monitoring dashboard.
        
        Args:
            collector: MetricsCollector instance
            port: Port to run dashboard on
        """
        self.logger = logging.getLogger(__name__)
        self.collector = collector
        self.port = port
        
        self.app = Dash(__name__)
        self._setup_layout()
        self._setup_callbacks()
        
        self.logger.info(f"Monitoring dashboard initialized on port {port}")
    
    def _setup_layout(self):
        """Set up dashboard layout."""
        self.app.layout = html.Div([
            html.H1("A-LMI System Monitoring Dashboard", 
                   style={'textAlign': 'center', 'marginBottom': '30px'}),
            
            # Kafka metrics
            html.Div([
                html.H2("Kafka Message Rates"),
                dcc.Graph(id='kafka-chart'),
                dcc.Interval(
                    id='interval-component',
                    interval=2*1000,  # Update every 2 seconds
                    n_intervals=0
                )
            ], style={'marginBottom': '30px'}),
            
            # Processing metrics
            html.Div([
                html.H2("Processing Throughput"),
                dcc.Graph(id='processing-chart')
            ], style={'marginBottom': '30px'}),
            
            # Memory usage
            html.Div([
                html.H2("Memory Usage"),
                dcc.Graph(id='memory-chart')
            ], style={'marginBottom': '30px'}),
            
            # Error rates
            html.Div([
                html.H2("Error Counts"),
                html.Div(id='error-counts')
            ], style={'marginBottom': '30px'}),
            
            # Hypotheses
            html.Div([
                html.H2("Hypothesis Generation"),
                dcc.Graph(id='hypotheses-chart')
            ])
        ])
    
    def _setup_callbacks(self):
        """Set up dashboard callbacks."""
        
        @self.app.callback(
            [Output('kafka-chart', 'figure'),
             Output('processing-chart', 'figure'),
             Output('memory-chart', 'figure'),
             Output('error-counts', 'children'),
             Output('hypotheses-chart', 'figure')],
            [Input('interval-component', 'n_intervals')]
        )
        def update_dashboard(n):
            """Update all dashboard components."""
            summary = self.collector.get_summary()
            
            # Kafka chart
            kafka_fig = self._create_kafka_chart(summary.get('kafka', {}))
            
            # Processing chart
            processing_fig = self._create_processing_chart(summary.get('processing', {}))
            
            # Memory chart
            memory_fig = self._create_memory_chart(summary.get('memory', {}))
            
            # Error counts
            error_div = self._create_error_counts(summary.get('errors', {}))
            
            # Hypotheses chart
            hypotheses_fig = self._create_hypotheses_chart(summary.get('hypotheses', {}))
            
            return kafka_fig, processing_fig, memory_fig, error_div, hypotheses_fig
    
    def _create_kafka_chart(self, data: Dict[str, Any]) -> go.Figure:
        """Create Kafka metrics chart."""
        topics = list(data.keys())
        counts = [data[t].get('count', 0) for t in topics]
        
        fig = go.Figure(data=go.Bar(x=topics, y=counts))
        fig.update_layout(
            title="Messages per Topic (Last 60 samples)",
            xaxis_title="Topic",
            yaxis_title="Count"
        )
        
        return fig
    
    def _create_processing_chart(self, data: Dict[str, Any]) -> go.Figure:
        """Create processing metrics chart."""
        services = list(data.keys())
        throughputs = [data[s].get('avg_throughput', 0) for s in services]
        
        fig = go.Figure(data=go.Bar(x=services, y=throughputs))
        fig.update_layout(
            title="Processing Throughput (items/sec)",
            xaxis_title="Service",
            yaxis_title="Throughput"
        )
        
        return fig
    
    def _create_memory_chart(self, data: Dict[str, Any]) -> go.Figure:
        """Create memory usage chart."""
        layers = list(data.keys())
        percents = [data[l].get('percent_used', 0) for l in layers]
        
        fig = go.Figure(data=go.Bar(x=layers, y=percents))
        fig.update_layout(
            title="Memory Usage by Layer",
            xaxis_title="Layer",
            yaxis_title="Percent Used",
            yaxis_range=[0, 100]
        )
        
        return fig
    
    def _create_error_counts(self, data: Dict[str, Any]) -> html.Div:
        """Create error counts display."""
        if not data:
            return html.Div("No errors recorded")
        
        items = []
        for key, count in data.items():
            items.append(html.P(f"{key}: {count}"))
        
        return html.Div(items)
    
    def _create_hypotheses_chart(self, data: Dict[str, Any]) -> go.Figure:
        """Create hypotheses chart."""
        categories = list(data.keys())
        counts = [data[c].get('count', 0) for c in categories]
        
        fig = go.Figure(data=go.Bar(x=categories, y=counts))
        fig.update_layout(
            title="Hypotheses Generated by Category",
            xaxis_title="Category",
            yaxis_title="Count"
        )
        
        return fig
    
    def run(self, debug: bool = False, threaded: bool = True):
        """
        Run the dashboard.
        
        Args:
            debug: Enable debug mode
            threaded: Run in separate thread
        """
        self.logger.info(f"Starting monitoring dashboard on port {self.port}")
        self.app.run_server(port=self.port, debug=debug, threaded=threaded)


def main():
    """Test dashboard."""
    from .metrics import MetricsCollector
    
    logging.basicConfig(level=logging.INFO)
    
    collector = MetricsCollector()
    
    # Add some sample data
    for i in range(50):
        collector.record_kafka_message('light_tokens', 1000 * (i % 10 + 1))
        collector.record_processing('processor', 100 + i * 2, 1)
    
    dashboard = MonitoringDashboard(collector)
    
    print(f"\nDashboard starting on http://localhost:{dashboard.port}")
    print("Press Ctrl+C to stop\n")
    
    dashboard.run(debug=True)


if __name__ == "__main__":
    import logging
    main()

