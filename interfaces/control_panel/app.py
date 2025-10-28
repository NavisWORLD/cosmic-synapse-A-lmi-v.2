"""
Cosmic Synapse Control Panel

Web-based interface for controlling the particle simulation parameters.
"""

import dash
from dash import dcc, html, Input, Output
import logging


class ControlPanel:
    """
    Web-based control panel for Cosmic Synapse simulation.
    
    Controls:
    - Ω (swirl strength)
    - λ (chaos parameter)
    - Particle count
    - Microphone toggle
    - Frequency presets
    """
    
    def __init__(self):
        """Initialize control panel."""
        self.logger = logging.getLogger(__name__)
        
        # Initialize Dash app
        self.app = dash.Dash(__name__, title="Cosmic Synapse Control Panel")
        
        # Current parameters
        self.current_params = {
            'omega': 0.1,
            'lambda': 0.05,
            'particle_count': 100,
            'mic_enabled': False
        }
        
        self._setup_layout()
        self._setup_callbacks()
        
        self.logger.info("Control panel initialized")
    
    def _setup_layout(self):
        """Setup control panel layout."""
        self.app.layout = html.Div([
            html.H1(
                "Cosmic Synapse Control Panel",
                style={'textAlign': 'center'}
            ),
            
            html.Div([
                # Parameter sliders
                html.Div([
                    html.Label('Swirl Strength (Ω)'),
                    dcc.Slider(
                        id='omega-slider',
                        min=0,
                        max=1,
                        step=0.01,
                        value=0.1,
                        marks={i: str(i) for i in [0, 0.2, 0.4, 0.6, 0.8, 1.0]}
                    )
                ], style={'margin': '20px'}),
                
                html.Div([
                    html.Label('Chaos Parameter (λ)'),
                    dcc.Slider(
                        id='lambda-slider',
                        min=0,
                        max=0.2,
                        step=0.01,
                        value=0.05,
                        marks={i: str(i) for i in [0, 0.05, 0.1, 0.15, 0.2]}
                    )
                ], style={'margin': '20px'}),
                
                html.Div([
                    html.Label('Particle Count'),
                    dcc.Slider(
                        id='particle-slider',
                        min=10,
                        max=1000,
                        step=10,
                        value=100,
                        marks={i: str(i) for i in [10, 250, 500, 750, 1000]}
                    )
                ], style={'margin': '20px'}),
                
                # Microphone toggle
                html.Div([
                    html.Label('Microphone Input'),
                    dcc.Checklist(
                        id='mic-toggle',
                        options=[{'label': 'Enable Microphone', 'value': 'enabled'}],
                        value=[]
                    )
                ], style={'margin': '20px'}),
                
                # Current values display
                html.Div(id='current-params', style={'margin': '20px', 'padding': '10px', 'backgroundColor': '#f0f0f0'})
            ])
        ])
    
    def _setup_callbacks(self):
        """Setup control panel callbacks."""
        @self.app.callback(
            Output('current-params', 'children'),
            [
                Input('omega-slider', 'value'),
                Input('lambda-slider', 'value'),
                Input('particle-slider', 'value'),
                Input('mic-toggle', 'value')
            ]
        )
        def update_params(omega, lambda_val, particle_count, mic_toggle):
            """Update current parameters display."""
            self.current_params['omega'] = omega
            self.current_params['lambda'] = lambda_val
            self.current_params['particle_count'] = particle_count
            self.current_params['mic_enabled'] = 'enabled' in mic_toggle
            
            return html.Div([
                html.H4("Current Parameters"),
                html.P(f"Ω: {omega:.3f}"),
                html.P(f"λ: {lambda_val:.3f}"),
                html.P(f"Particles: {particle_count}"),
                html.P(f"Microphone: {'ON' if self.current_params['mic_enabled'] else 'OFF'}")
            ])
    
    def run(self, host='127.0.0.1', port=8051, debug=True):
        """Run control panel web interface."""
        self.logger.info(f"Starting control panel on http://{host}:{port}")
        self.app.run_server(host=host, port=port, debug=debug)


def main():
    """Run control panel."""
    panel = ControlPanel()
    panel.run()


if __name__ == "__main__":
    main()

