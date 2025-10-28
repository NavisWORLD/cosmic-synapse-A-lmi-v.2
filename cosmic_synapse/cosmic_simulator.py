"""
Cosmic Synapse Simulation - Python Visualization

Real-time particle simulation with audio-driven stochastic resonance.
Implements the vibrational information dynamics from the 8D equation.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
import pyaudio
from scipy.fft import fft
import threading
import queue
import time


class CosmicSimulation:
    """
    Particle simulation implementing vibrational information dynamics.
    
    Implements:
    - Conservative bowl potential
    - Gravitational fields
    - Swirl forces
    - Audio-driven noise modulation (stochastic resonance)
    """
    
    def __init__(self, n_particles=100):
        """
        Initialize simulation.
        
        Args:
            n_particles: Number of particles
        """
        self.n_particles = n_particles
        self.config = self._load_config()
        
        # Initialize particles (random positions and velocities)
        self.positions = np.random.randn(n_particles, 2) * 2.0
        self.velocities = np.zeros_like(self.positions)
        
        # Physics parameters
        self.dt = self.config['cosmic_synapse']['physics']['timestep']
        self.alpha_phi = self.config['cosmic_synapse']['physics']['alpha_phi']
        self.beta = self.config['cosmic_synapse']['physics']['beta']
        self.gamma = self.config['cosmic_synapse']['physics']['gamma']
        self.kappa_omega = self.config['cosmic_synapse']['physics']['kappa_omega']
        self.zeta = self.config['cosmic_synapse']['physics']['zeta']
        
        # Noise parameters
        self.sigma_lambda = self.config['cosmic_synapse']['physics']['sigma_lambda_base']
        self.sigma_psd_scale = self.config['cosmic_synapse']['physics']['sigma_psd_scale']
        
        # Audio parameters
        self.enable_audio = False
        self.audio_queue = queue.Queue()
        self.psd_normalized = 0.0
        
        # Figure setup
        self.fig, self.ax = plt.subplots(figsize=(12, 12))
        self.ax.set_xlim(-5, 5)
        self.ax.set_ylim(-5, 5)
        self.ax.set_aspect('equal')
        self.ax.set_title('Cosmic Synapse: Vibrational Information Dynamics')
        
        self.scatter = None
        
        # Control parameters
        self.omega = 0.1  # Swirl strength
        self.lambda_val = 0.05  # Chaos parameter
        self.mic_enabled = False
    
    def _load_config(self):
        """Load configuration."""
        import yaml
        try:
            with open('../infrastructure/config.yaml', 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            # Return defaults
            return {
                'cosmic_synapse': {
                    'physics': {
                        'timestep': 0.01,
                        'alpha_phi': 1.0,
                        'beta': 0.5,
                        'gamma': 0.3,
                        'kappa_omega': 0.2,
                        'zeta': 0.05,
                        'sigma_lambda_base': 0.1,
                        'sigma_psd_scale': 0.5
                    }
                }
            }
    
    def compute_forces(self):
        """
        Compute all forces acting on particles.
        
        Implements:
        - Conservative bowl: φ(x) = (1/2)||x-x₀||²
        - Swirl: F_swirl = κ_Ω·Ω·R_π/2(x - x₀)
        - Damping: F_damp = -ζu
        - Noise: σξ(τ) with sigma from audio or lambda
        """
        # Bowl potential (conservative, radial)
        x0 = np.array([0.0, 0.0])
        bowl_force = -self.alpha_phi * (self.positions - x0)
        
        # Swirl force (perpendicular to radius)
        radius = self.positions - x0
        swirl_direction = np.array([-radius[:, 1], radius[:, 0]]).T
        swirl_force = self.kappa_omega * self.omega * swirl_direction
        
        # Damping
        damping_force = -self.zeta * self.velocities
        
        # Noise (stochastic resonance)
        # If mic enabled, use PSD; otherwise use lambda
        if self.mic_enabled:
            sigma = self.sigma_psd_scale * self.psd_normalized
        else:
            sigma = self.sigma_lambda * self.lambda_val
        
        noise_force = sigma * np.random.randn(self.n_particles, 2)
        
        # Total force
        total_force = bowl_force + swirl_force + damping_force + noise_force
        
        return total_force
    
    def update_step(self):
        """Update particle positions and velocities."""
        # Compute forces
        forces = self.compute_forces()
        
        # Update velocities: du/dτ = F - ζu (from compute_forces)
        self.velocities += self.dt * forces
        
        # Update positions: dx/dτ = u
        self.positions += self.dt * self.velocities
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Audio stream callback."""
        if self.mic_enabled:
            self.audio_queue.put(in_data)
        return (None, pyaudio.paContinue)
    
    def _compute_psd(self):
        """Compute normalized PSD from audio."""
        if not self.audio_queue.empty():
            try:
                audio_data = self.audio_queue.get_nowait()
                audio_array = np.frombuffer(audio_data, dtype=np.int16)
                
                # Compute FFT
                fft_result = fft(audio_array)
                power = np.abs(fft_result) ** 2
                
                # Normalize to [0, 1]
                self.psd_normalized = float(np.mean(power) / (np.max(power) + 1e-10))
                
            except queue.Empty:
                pass
    
    def animate(self, frame):
        """
        Animation function for matplotlib.
        
        Args:
            frame: Frame number
        """
        # Update audio PSD if enabled
        if self.mic_enabled:
            self._compute_psd()
        
        # Physics update
        self.update_step()
        
        # Update visualization
        self.ax.clear()
        self.ax.set_xlim(-5, 5)
        self.ax.set_ylim(-5, 5)
        self.ax.set_aspect('equal')
        self.ax.set_title(f'Cosmic Synapse - Frame {frame}')
        
        # Plot particles
        self.ax.scatter(
            self.positions[:, 0],
            self.positions[:, 1],
            c=np.arange(self.n_particles),
            cmap='hsv',
            s=50,
            alpha=0.6
        )
        
        # Plot center
        self.ax.scatter(0, 0, c='red', s=100, marker='*')
        
        # Display parameters
        info_text = f'Ω={self.omega:.2f} λ={self.lambda_val:.2f}'
        if self.mic_enabled:
            info_text += f' PSD={self.psd_normalized:.3f}'
        self.ax.text(-4.5, 4, info_text, fontsize=12)
    
    def run(self):
        """Run simulation with animation."""
        self.anim = animation.FuncAnimation(
            self.fig,
            self.animate,
            interval=16,  # ~60 FPS
            cache_frame_data=False
        )
        plt.show()
    
    def start_audio(self):
        """Start audio input."""
        self.mic_enabled = True
        self.audio = pyaudio.PyAudio()
        
        self.stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=44100,
            frames_per_buffer=4096,
            input=True,
            stream_callback=self._audio_callback
        )
        
        self.stream.start_stream()
        print("Microphone enabled - PSD modulation active")
    
    def stop_audio(self):
        """Stop audio input."""
        self.mic_enabled = False
        if hasattr(self, 'stream'):
            self.stream.stop_stream()
            self.stream.close()
        if hasattr(self, 'audio'):
            self.audio.terminate()
        print("Microphone disabled")


def main():
    """Run simulation."""
    sim = CosmicSimulation(n_particles=100)
    
    # Uncomment to enable microphone
    # sim.start_audio()
    
    try:
        print("Starting simulation...")
        print("Press Ctrl+C to stop")
        sim.run()
    except KeyboardInterrupt:
        print("\nStopping simulation...")
        sim.stop_audio()


if __name__ == "__main__":
    main()

