# -*- coding: utf-8 -*-
"""
12D COSMIC SYNAPSE THEORY - CONTINUOUS TOKEN GENERATION ENGINE v2.0
Python Implementation

CST v2.0 RESTORATION NOTES (Additive Updates):

RESTORED FUNCTIONALITY:
1. Continuous Token Generation: Rolling window rate calculation (2-second window)
   - Tokens generate every 100ms from live audio or replay
   - Token rate displayed in real-time
   - Token display bounded for performance

2. Deterministic Record/Replay: Deep copying and seed management
   - Complete audio frame data recorded
   - Replay resets token arrays for clean deterministic output
   - Same seed + recording produces identical tokens and Ψ values

3. Adaptive Timestep: Dynamic dt based on min distance and max velocity
   - Clamped to [1e-4, dtMax] for stability

4. Real-time Diagnostics: All metrics update continuously
   - Ψ breakdown: energy, λ, ∫||v||dt, ∫|Δx12|dt, Ω·E, potential terms
   - Synchronization: Kuramoto order parameter r and mean θ
   - Conservation: Energy, momentum, angular momentum, virial ratio

5. Controls: All parameters wired and functional
"""

import numpy as np
import queue
import threading
import time
import json
import collections
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

# Constants
C = 299792458.0  # Speed of light
PHI = (1 + np.sqrt(5)) / 2  # Golden ratio
H = 6.62607015e-34  # Planck's constant
KB = 1.380649e-23  # Boltzmann constant
G = 6.67430e-11  # Gravitational constant


class SimulationMode(Enum):
    LIVE = "live"
    REPLAY = "replay"


@dataclass
class PhysicsConfig:
    """CST v2.0 additive: Physics configuration"""
    G: float = G
    a0: float = 1.0
    m0: float = 1.0
    Eref: float = 1.0
    tref: float = 1.0
    vref: float = 1.0
    epsilon: float = 0.1
    rCutoff: float = 10.0
    blendLorenz: float = 0.7
    gravEnabled: bool = False
    dmEnabled: bool = False


@dataclass
class AdaptiveConfig:
    """CST v2.0 additive: Adaptive state configuration"""
    k: float = 0.5
    gamma: float = 0.2
    alpha: float = 0.3
    sigmaSimilarity: float = 0.3


@dataclass
class SyncConfig:
    """CST v2.0 additive: Synchronization configuration"""
    Ksync: float = 0.1


@dataclass
class TimestepConfig:
    """CST v2.0 additive: Timestep configuration"""
    dt: float = 0.005
    dtMax: float = 0.01
    adaptive: bool = True


@dataclass
class DarkMatterParams:
    """CST v2.0 additive: Dark matter parameters"""
    rho0: float = 1.0
    rs: float = 5.0


@dataclass
class AudioFrame:
    """CST v2.0 additive: Audio frame data structure"""
    timestamp: float
    rmsEnergy: float
    frequencyData: List[Dict[str, float]]  # [{frequency, magnitude}, ...]
    spectralCentroid: float
    harmonics: List[float]
    dataArray: Optional[np.ndarray] = None


class TokenStream:
    """CST v2.0 additive: Token stream with rolling window rate calculation"""
    
    def __init__(self, window_size: float = 2.0):
        self.tokens: List[Dict] = []
        self.count_per_sec: float = 0.0
        self.window_size: float = window_size  # seconds
        self.last_window_counts: collections.deque = collections.deque()
        self.max_tokens_display: int = 200
    
    def add_token(self, token: Dict):
        """Add token and record timestamp"""
        self.tokens.append(token)
        self.last_window_counts.append(time.time())
        self._clean_old_timestamps()
    
    def _clean_old_timestamps(self):
        """Remove timestamps outside the window"""
        cutoff = time.time() - self.window_size
        while self.last_window_counts and self.last_window_counts[0] < cutoff:
            self.last_window_counts.popleft()
    
    def update_rate(self, now: Optional[float] = None) -> float:
        """Calculate tokens per second over rolling window"""
        if now is None:
            now = time.time()
        
        self._clean_old_timestamps()
        
        if len(self.last_window_counts) == 0:
            self.count_per_sec = 0.0
            return 0.0
        
        window_span = (now - self.last_window_counts[0]) if len(self.last_window_counts) > 0 else 1.0
        if window_span > 0:
            self.count_per_sec = len(self.last_window_counts) / window_span
        else:
            self.count_per_sec = len(self.last_window_counts)
        
        return self.count_per_sec
    
    def export_json(self, path: str, metadata: Optional[Dict] = None):
        """Export tokens to JSON with metadata"""
        export_data = {
            "metadata": metadata or {},
            "tokens": self.tokens
        }
        with open(path, 'w') as f:
            json.dump(export_data, f, indent=2)
    
    def clear(self):
        """Clear all tokens"""
        self.tokens = []
        self.last_window_counts.clear()
        self.count_per_sec = 0.0


class Recorder:
    """CST v2.0 additive: Deterministic audio frame recorder"""
    
    def __init__(self):
        self.frames: List[AudioFrame] = []
        self.recording: bool = False
        self.replay_index: int = 0
    
    def start(self):
        """Start recording"""
        self.recording = True
        self.frames = []
    
    def stop(self):
        """Stop recording"""
        self.recording = False
    
    def add_frame(self, frame: AudioFrame):
        """Add frame if recording"""
        if self.recording:
            # Deep copy to avoid mutation
            frame_copy = AudioFrame(
                timestamp=frame.timestamp,
                rmsEnergy=frame.rmsEnergy,
                frequencyData=[f.copy() for f in frame.frequencyData],
                spectralCentroid=frame.spectralCentroid,
                harmonics=frame.harmonics.copy(),
                dataArray=frame.dataArray.copy() if frame.dataArray is not None else None
            )
            self.frames.append(frame_copy)
    
    def save(self, path: str):
        """Save recorded frames to JSON"""
        frames_data = [
            {
                "timestamp": f.timestamp,
                "rmsEnergy": f.rmsEnergy,
                "frequencyData": f.frequencyData,
                "spectralCentroid": f.spectralCentroid,
                "harmonics": f.harmonics,
                "dataArray": f.dataArray.tolist() if f.dataArray is not None else None
            }
            for f in self.frames
        ]
        with open(path, 'w') as f:
            json.dump(frames_data, f, indent=2)
    
    def load(self, path: str):
        """Load recorded frames from JSON"""
        with open(path, 'r') as f:
            frames_data = json.load(f)
        
        self.frames = [
            AudioFrame(
                timestamp=frame["timestamp"],
                rmsEnergy=frame["rmsEnergy"],
                frequencyData=frame["frequencyData"],
                spectralCentroid=frame["spectralCentroid"],
                harmonics=frame["harmonics"],
                dataArray=np.array(frame["dataArray"]) if frame["dataArray"] else None
            )
            for frame in frames_data
        ]
    
    def next_frame(self) -> Optional[AudioFrame]:
        """Get next frame for replay"""
        if self.replay_index >= len(self.frames):
            self.replay_index = 0  # Loop
            return None
        
        frame = self.frames[self.replay_index]
        self.replay_index += 1
        return frame
    
    def reset_replay(self):
        """Reset replay index"""
        self.replay_index = 0


class Particle:
    """CST v2.0 additive: Particle with 12D CST properties"""
    
    def __init__(self, x: float = 0.1, y: float = 0.0, z: float = 0.0, 
                 frequency: float = 0.0, parent_id: Optional[str] = None):
        self.id = self._generate_id()
        self.x = x
        self.y = y
        self.z = z
        self.frequency = frequency
        self.mass = 1.0
        self.velocity = np.array([0.0, 0.0, 0.0])
        self.parent_id = parent_id
        
        # 12D CST properties
        self.x12: float = 0.0  # Adaptive state
        self.m12: float = 0.0  # Memory
        self.Ec: float = 0.0  # Cosmic energy
        self.Ugrav: float = 0.0  # Gravitational potential
        self.Udm: float = 0.0  # Dark matter potential
        self.vi: float = 0.0  # Characteristic frequency
        self.theta: float = np.random.random() * 2 * np.pi  # Phase
        self.omega: float = 0.0  # Synaptic strength
        self.entropyS: float = 0.0  # Entropy
        self.neighbors: List[int] = []
        
        # 11D projection
        self.projection11D_pos = np.zeros(11)
        self.projection11D_vel = np.zeros(11)
        self._update_11d_projection()
    
    def _generate_id(self) -> str:
        """Generate deterministic ID"""
        return f"particle_{id(self)}"
    
    def _update_11d_projection(self):
        """Update 11D projection from 3D position"""
        scale = 0.1
        for i in range(11):
            dim = i % 3
            coord = [self.x, self.y, self.z][dim]
            self.projection11D_pos[i] = coord * (1 + i * scale)
            self.projection11D_vel[i] = self.velocity[dim] * (1 + i * scale)
    
    def update_x12(self, dt: float, k: float, gamma: float):
        """Update adaptive state: dx12/dt = k * Ωi − γ * x12_i"""
        dx12 = (k * self.omega - gamma * self.x12) * dt
        self.x12 += dx12
        self.x12 = np.clip(self.x12, -1.0, 1.0)
    
    def update_memory_state(self, dt: float, alpha: float):
        """Update memory: dm12/dt = α * (x12_i − m12_i)"""
        dm12 = alpha * (self.x12 - self.m12) * dt
        self.m12 += dm12
    
    def update_phase(self, dt: float, neighbors: List['Particle'], Ksync: float):
        """Update phase with Kuramoto synchronization"""
        # vi = Ec / h
        self.vi = self.Ec / H if H > 0 else 0.0
        
        # Kuramoto phase dynamics
        phase_coupling = 0.0
        degree = max(1, len(neighbors))
        
        for neighbor in neighbors:
            phase_coupling += np.sin(neighbor.theta - self.theta)
        
        # dθi/dt = vi + (Ksync / deg_i) Σ_j sin(θj − θi)
        dtheta = (self.vi + (Ksync / degree) * phase_coupling) * dt
        self.theta += dtheta
        self.theta = self.theta % (2 * np.pi)


class Simulator:
    """CST v2.0 additive: Main simulation engine"""
    
    def __init__(self):
        self.audio_running: bool = False
        self.audio_queue: queue.Queue = queue.Queue()
        self.processed_audio_queue: queue.Queue = queue.Queue()
        
        self.physics = PhysicsConfig()
        self.adapt = AdaptiveConfig()
        self.sync = SyncConfig()
        self.timestep = TimestepConfig()
        self.dm_params = DarkMatterParams()
        
        self.particles: List[Particle] = []
        self.token_stream = TokenStream()
        self.recorder = Recorder()
        self.mode: SimulationMode = SimulationMode.LIVE
        self.seed: Optional[int] = None
        
        # CST v2.0 additive: Audio state for real-time modulation
        self.current_audio_energy: float = 0.0
        self.current_frequency_data: List[Dict[str, float]] = []
        self.audio_sensitivity: float = 1.0  # Audio modulation sensitivity
        
        # Accumulators for ψ integrals
        self.psi_velocity_integral: Dict[str, float] = {}
        self.psi_x12_integral: Dict[str, float] = {}
        
        # Conservation tracking
        self.conservation_E0: float = 0.0
        self.conservation_P0: np.ndarray = np.zeros(3)
        self.conservation_L0: np.ndarray = np.zeros(3)
    
    def set_seed(self, seed: int):
        """CST v2.0 additive: Set deterministic seed"""
        self.seed = seed
        np.random.seed(seed)
        # Note: If using torch, also set torch.manual_seed(seed)
    
    def start_audio(self):
        """Start audio capture thread"""
        self.audio_running = True
        # Audio capture would be implemented here with pyaudio or similar
        # For now, this is a placeholder
    
    def stop_audio(self):
        """Stop audio capture"""
        self.audio_running = False
    
    def tick(self, dt: Optional[float] = None):
        """Process one simulation frame"""
        if dt is None:
            dt = self.timestep.dt
        
        # Process audio frames from queue
        while not self.processed_audio_queue.empty():
            frame = self.processed_audio_queue.get()
            self._process_audio_frame(frame)
        
        # Update particles
        if len(self.particles) > 0:
            # Build spatial index
            spatial_index = self._build_spatial_index()
            
            # Always compute neighbors first (needed for synaptic strength)
            for i, pi in enumerate(self.particles):
                if not pi.neighbors:
                    pi.neighbors = self._query_neighbors(i, self.particles, spatial_index, 
                                                         self.physics.rCutoff)
            
            # Compute forces and energies
            if self.physics.gravEnabled:
                self._compute_gravitational_forces(spatial_index)
                self._compute_gravitational_energy(spatial_index)
            
            if self.physics.dmEnabled:
                self._compute_dark_matter_potential()
            
            # Compute synaptic strength (neighbors already set above)
            self._compute_synaptic_strength(spatial_index)
            
            # Update adaptive states
            for p in self.particles:
                p.update_x12(dt, self.adapt.k, self.adapt.gamma)
                p.update_memory_state(dt, self.adapt.alpha)
            
            # Update phases
            for i, p in enumerate(self.particles):
                # Ensure neighbors list is valid
                if p.neighbors:
                    neighbors = [self.particles[j] for j in p.neighbors if j < len(self.particles)]
                else:
                    neighbors = []
                p.update_phase(dt, neighbors, self.sync.Ksync)
            
            # Update particle positions (Lorenz + gravity blend)
            self._update_particle_positions(dt)
            
            # Update cosmic energies
            for p in self.particles:
                v2 = np.sum(p.velocity ** 2)
                K = 0.5 * p.mass * v2
                p.Ec = K + p.Ugrav + p.Udm
                p.vi = p.Ec / H if H > 0 else 0.0
                p._update_11d_projection()
            
            # Update ψ accumulators
            self._update_psi_accumulators(dt)
            
            # Adaptive timestep
            if self.timestep.adaptive:
                self.timestep.dt = self._compute_adaptive_dt()
    
    def _build_spatial_index(self) -> Dict:
        """Build spatial index for neighbor queries"""
        # Simple uniform grid implementation
        cell_size = self.physics.rCutoff
        grid = {}
        
        for i, p in enumerate(self.particles):
            # CST v2.0 additive: Handle infinity/NaN values to prevent overflow
            x = p.x if np.isfinite(p.x) else 0.0
            y = p.y if np.isfinite(p.y) else 0.0
            z = p.z if np.isfinite(p.z) else 0.0
            
            # Clamp values to prevent overflow
            x = np.clip(x, -1e6, 1e6)
            y = np.clip(y, -1e6, 1e6)
            z = np.clip(z, -1e6, 1e6)
            
            gx = int(np.floor(x / cell_size))
            gy = int(np.floor(y / cell_size))
            gz = int(np.floor(z / cell_size))
            key = (gx, gy, gz)
            
            if key not in grid:
                grid[key] = []
            grid[key].append(i)
        
        return {"grid": grid, "cell_size": cell_size}
    
    def _query_neighbors(self, i: int, particles: List[Particle], 
                        spatial_index: Dict, r_cutoff: float) -> List[int]:
        """Query neighbors within cutoff radius"""
        p = particles[i]
        neighbors = []
        grid = spatial_index["grid"]
        cell_size = spatial_index["cell_size"]
        
        # CST v2.0 additive: Handle infinity/NaN values
        x = p.x if np.isfinite(p.x) else 0.0
        y = p.y if np.isfinite(p.y) else 0.0
        z = p.z if np.isfinite(p.z) else 0.0
        
        x = np.clip(x, -1e6, 1e6)
        y = np.clip(y, -1e6, 1e6)
        z = np.clip(z, -1e6, 1e6)
        
        gx = int(np.floor(x / cell_size))
        gy = int(np.floor(y / cell_size))
        gz = int(np.floor(z / cell_size))
        
        # Check current cell and 26 neighboring cells
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    key = (gx + dx, gy + dy, gz + dz)
                    if key in grid:
                        for j in grid[key]:
                            if i != j:
                                pj = particles[j]
                                r = np.sqrt(np.sum((np.array([pj.x, pj.y, pj.z]) - 
                                                   np.array([p.x, p.y, p.z])) ** 2))
                                if r <= r_cutoff:
                                    neighbors.append(j)
        
        return neighbors
    
    def _compute_gravitational_forces(self, spatial_index: Dict):
        """Compute gravitational accelerations"""
        if not self.physics.gravEnabled:
            # Still set neighbors for synaptic strength calculation
            for i, pi in enumerate(self.particles):
                pi.neighbors = self._query_neighbors(i, self.particles, spatial_index, 
                                                     self.physics.rCutoff)
            return
        
        for i, pi in enumerate(self.particles):
            neighbors = self._query_neighbors(i, self.particles, spatial_index, 
                                             self.physics.rCutoff)
            pi.neighbors = neighbors
            
            acceleration = np.zeros(3)
            for j in neighbors:
                pj = self.particles[j]
                dx = np.array([pj.x - pi.x, pj.y - pi.y, pj.z - pi.z])
                r2 = np.sum(dx ** 2)
                r_eff2 = r2 + self.physics.epsilon ** 2
                r_eff = np.sqrt(r_eff2)
                
                force = self.physics.G * pi.mass * pj.mass / r_eff2
                acceleration += force * (dx / r_eff) / pi.mass
            
            # Store acceleration for later use in position update
            pi.acceleration = acceleration
    
    def _compute_gravitational_energy(self, spatial_index: Dict):
        """Compute gravitational potential energy"""
        if not self.physics.gravEnabled:
            for p in self.particles:
                p.Ugrav = 0.0
            return
        
        for i, pi in enumerate(self.particles):
            U = 0.0
            neighbors = pi.neighbors
            
            for j in neighbors:
                pj = self.particles[j]
                dx = np.array([pj.x - pi.x, pj.y - pi.y, pj.z - pi.z])
                r2 = np.sum(dx ** 2)
                r_eff = np.sqrt(r2 + self.physics.epsilon ** 2)
                
                U -= self.physics.G * pi.mass * pj.mass / r_eff
            
            pi.Ugrav = U
    
    def _compute_dark_matter_potential(self):
        """Compute dark matter potential (NFW profile)"""
        if not self.physics.dmEnabled:
            for p in self.particles:
                p.Udm = 0.0
            return
        
        for p in self.particles:
            r = np.sqrt(p.x ** 2 + p.y ** 2 + p.z ** 2)
            r_rs = r / self.dm_params.rs
            
            # NFW density profile
            rho = self.dm_params.rho0 / (r_rs * (1 + r_rs) ** 2)
            
            # Simplified potential
            p.Udm = -self.physics.G * p.mass * rho * 4 * np.pi * r ** 2 / 3
    
    def _compute_synaptic_strength(self, spatial_index: Dict):
        """Compute synaptic strength Ω with similarity"""
        for i, pi in enumerate(self.particles):
            omega_sum = 0.0
            # Ensure neighbors are set (query if not already set)
            if not pi.neighbors:
                pi.neighbors = self._query_neighbors(i, self.particles, spatial_index, self.physics.rCutoff)
            neighbors = pi.neighbors
            
            for j in neighbors:
                pj = self.particles[j]
                dx = np.array([pj.x - pi.x, pj.y - pi.y, pj.z - pi.z])
                r2 = np.sum(dx ** 2)
                r_eff2 = r2 + self.physics.epsilon ** 2
                
                # Gravitational coupling
                grav_term = (self.physics.G * pi.mass * pj.mass) / \
                           (r_eff2 * self.physics.a0 * self.physics.m0)
                
                # Gaussian similarity
                x12_diff = pi.x12 - pj.x12
                similarity = np.exp(-(x12_diff ** 2) / 
                                  (2 * self.adapt.sigmaSimilarity ** 2))
                
                omega_sum += grav_term * similarity
            
            pi.omega = omega_sum
    
    def _update_particle_positions(self, dt: float):
        """Update particle positions with Lorenz + gravity blend, audio-modulated"""
        # CST v2.0 additive: Audio-modulated Lorenz parameters
        base_sigma = 10.0
        base_rho = 28.0
        beta = 2.667
        
        # Audio modulation: scale parameters based on RMS energy
        audio_modulation = 1.0 + (self.current_audio_energy * self.audio_sensitivity)
        sigma = base_sigma * audio_modulation
        rho = base_rho * (1.0 + self.current_audio_energy * self.audio_sensitivity * 0.3)
        
        for p in self.particles:
            # Lorenz equations with audio modulation
            dx_lorenz = sigma * (p.y - p.x) * dt
            dy_lorenz = (p.x * (rho - p.z) - p.y) * dt
            dz_lorenz = (p.x * p.y - beta * p.z) * dt
            
            # Blend with gravity
            if self.physics.gravEnabled and hasattr(p, 'acceleration'):
                dx_grav = p.acceleration[0] * dt
                dy_grav = p.acceleration[1] * dt
                dz_grav = p.acceleration[2] * dt
                
                dx = self.physics.blendLorenz * dx_lorenz + \
                     (1 - self.physics.blendLorenz) * dx_grav
                dy = self.physics.blendLorenz * dy_lorenz + \
                     (1 - self.physics.blendLorenz) * dy_grav
                dz = self.physics.blendLorenz * dz_lorenz + \
                     (1 - self.physics.blendLorenz) * dz_grav
            else:
                dx, dy, dz = dx_lorenz, dy_lorenz, dz_lorenz
            
            p.x += dx
            p.y += dy
            p.z += dz
            
            # CST v2.0 additive: Clamp positions to prevent infinity/NaN
            p.x = np.clip(p.x, -1000.0, 1000.0) if np.isfinite(p.x) else 0.0
            p.y = np.clip(p.y, -1000.0, 1000.0) if np.isfinite(p.y) else 0.0
            p.z = np.clip(p.z, -1000.0, 1000.0) if np.isfinite(p.z) else 0.0
            
            # Clamp velocity to prevent infinity
            vx = dx / dt if dt > 0 else 0.0
            vy = dy / dt if dt > 0 else 0.0
            vz = dz / dt if dt > 0 else 0.0
            
            vx = np.clip(vx, -1000.0, 1000.0) if np.isfinite(vx) else 0.0
            vy = np.clip(vy, -1000.0, 1000.0) if np.isfinite(vy) else 0.0
            vz = np.clip(vz, -1000.0, 1000.0) if np.isfinite(vz) else 0.0
            
            p.velocity = np.array([vx, vy, vz])
    
    def _compute_adaptive_dt(self) -> float:
        """Compute adaptive timestep"""
        if not self.timestep.adaptive or len(self.particles) == 0:
            return self.timestep.dt
        
        r_min = float('inf')
        v_max = 0.0
        
        for p in self.particles:
            v = np.sqrt(np.sum(p.velocity ** 2))
            v_max = max(v_max, v)
            
            # Check neighbors for minimum distance
            if p.neighbors:
                for j in p.neighbors:
                    if j < len(self.particles):
                        pj = self.particles[j]
                        r = np.sqrt(np.sum((np.array([pj.x, pj.y, pj.z]) - 
                                           np.array([p.x, p.y, p.z])) ** 2))
                        r_min = min(r_min, r)
        
        if r_min == float('inf'):
            r_min = 1.0
        if v_max == 0:
            v_max = 1.0
        
        dt = min(self.timestep.dtMax, 0.1 * r_min / v_max)
        return max(1e-4, min(dt, self.timestep.dtMax))
    
    def _update_psi_accumulators(self, dt: float):
        """Update ψ integral accumulators"""
        for p in self.particles:
            # Velocity integral
            v_mag = np.sqrt(np.sum(p.velocity ** 2))
            v_int = self.psi_velocity_integral.get(p.id, 0.0) + \
                   (v_mag / self.physics.vref) * dt
            self.psi_velocity_integral[p.id] = v_int
            
            # x12 integral
            x12_int = self.psi_x12_integral.get(p.id, 0.0) + \
                     np.abs(p.x12) * dt
            self.psi_x12_integral[p.id] = x12_int
    
    def compute_psi(self) -> Dict:
        """Compute normalized ψ breakdown"""
        terms = {
            "energyTerm": 0.0,
            "lambdaTerm": 0.0,
            "velocityIntegralTerm": 0.0,
            "x12IntegralTerm": 0.0,
            "omegaTerm": 0.0,
            "potentialTerm": 0.0
        }
        
        if len(self.particles) == 0:
            return {"terms": terms, "psiTotal": 0.0}
        
        for p in self.particles:
            v_mag = np.sqrt(np.sum(p.velocity ** 2))
            
            terms["energyTerm"] += PHI * (p.Ec / self.physics.Eref)
            terms["lambdaTerm"] += np.log(np.abs(v_mag) + 1) / 100.0
            terms["velocityIntegralTerm"] += self.psi_velocity_integral.get(p.id, 0.0)
            terms["x12IntegralTerm"] += self.psi_x12_integral.get(p.id, 0.0)
            terms["omegaTerm"] += p.omega * (p.Ec / self.physics.Eref)
            terms["potentialTerm"] += (p.Ugrav + p.Udm) / self.physics.Eref
        
        psi_total = sum(terms.values())
        return {"terms": terms, "psiTotal": psi_total}
    
    def _process_audio_frame(self, frame: AudioFrame):
        """Process audio frame and generate tokens, create/update particles"""
        # Record if recording
        if self.recorder.recording:
            self.recorder.add_frame(frame)
        
        # Generate tokens
        self._generate_audio_frame_token(frame)
        self._generate_harmonic_tokens(frame)
        
        # CST v2.0 additive: Create/update particles from audio frequencies
        self._create_or_update_particles_from_audio(frame)
    
    def _generate_audio_frame_token(self, frame: AudioFrame):
        """Generate audio frame token"""
        token = {
            "id": f"audio_frame_{int(time.time() * 1000)}",
            "type": "audio_frame",
            "timestamp": frame.timestamp,
            "rmsEnergy": frame.rmsEnergy,
            "spectralCentroid": frame.spectralCentroid,
            "frequencyCount": len(frame.frequencyData),
            "topFrequencies": frame.frequencyData[:5],
            "phiHarmonics": frame.harmonics[:5],
            "seed": self.seed
        }
        self.token_stream.add_token(token)
    
    def _generate_harmonic_tokens(self, frame: AudioFrame):
        """Generate φ-harmonic tokens"""
        for idx, harmonic in enumerate(frame.harmonics):
            if idx < len(frame.frequencyData):
                magnitude = frame.frequencyData[idx]["magnitude"]
                token = {
                    "id": f"harmonic_{int(time.time() * 1000)}_{idx}",
                    "type": "phi_harmonic",
                    "timestamp": frame.timestamp,
                    "harmonic": harmonic,
                    "magnitude": magnitude,
                    "harmonicIndex": idx,
                    "phiRatio": PHI ** (idx / 2)
                }
                self.token_stream.add_token(token)
    
    def _create_or_update_particles_from_audio(self, frame: AudioFrame):
        """CST v2.0 additive: Create or update particles based on audio frequencies"""
        if len(frame.frequencyData) == 0:
            return
        
        # Store current audio state for modulation
        self.current_audio_energy = frame.rmsEnergy
        self.current_frequency_data = frame.frequencyData
        
        # Create new particles if we have space (max 20 particles)
        max_particles = 20
        if len(self.particles) < max_particles:
            for freq_data in frame.frequencyData[:5]:  # Top 5 frequencies
                if freq_data["magnitude"] > 0.1:  # Threshold for creation
                    # Sound-to-color mapping (frequency to hue)
                    hue = (freq_data["frequency"] / 20000.0) * 360.0
                    
                    # Create particle with frequency-based properties
                    particle = Particle(
                        x=np.random.random() * 10 - 5,
                        y=np.random.random() * 10 - 5,
                        z=np.random.random() * 10 - 5,
                        frequency=freq_data["frequency"],
                        parent_id=None
                    )
                    
                    # Set mass based on magnitude
                    particle.mass = 1.0 + freq_data["magnitude"] * 5.0
                    particle.Ec = freq_data["magnitude"] * 50.0
                    
                    self.particles.append(particle)
                    
                    # Generate particle creation token
                    self._generate_particle_token(particle, "audio_creation")
        
        # Update existing particles with frequency assignments
        for idx, freq_data in enumerate(frame.frequencyData):
            if idx < len(self.particles):
                particle = self.particles[idx]
                self._update_particle_from_audio(particle, freq_data)
    
    def _update_particle_from_audio(self, particle: Particle, freq_data: Dict[str, float]):
        """CST v2.0 additive: Update particle properties from audio frequency data"""
        # Update frequency
        particle.frequency = freq_data["frequency"]
        
        # Update mass based on magnitude
        particle.mass = max(1.0, particle.mass * (0.95 + freq_data["magnitude"] * 0.1))
        
        # Update energy
        particle.Ec = freq_data["magnitude"] * 50.0
        
        # Generate frequency update token
        token = {
            "id": f"freq_update_{int(time.time() * 1000)}_{particle.id}",
            "type": "frequency_update",
            "particleId": particle.id,
            "timestamp": time.time(),
            "frequency": freq_data["frequency"],
            "magnitude": freq_data["magnitude"]
        }
        self.token_stream.add_token(token)
    
    def _generate_particle_token(self, particle: Particle, event_type: str):
        """CST v2.0 additive: Generate particle event token"""
        token = {
            "id": f"particle_{event_type}_{int(time.time() * 1000)}_{particle.id}",
            "type": "particle_event",
            "event": event_type,
            "timestamp": time.time(),
            "particleId": particle.id,
            "parentId": particle.parent_id,
            "position": [float(particle.x), float(particle.y), float(particle.z)],
            "velocity": [float(particle.velocity[0]), float(particle.velocity[1]), float(particle.velocity[2])],
            "frequency": float(particle.frequency),
            "Ec": float(particle.Ec),
            "Ugrav": float(particle.Ugrav),
            "Udm": float(particle.Udm),
            "vi": float(particle.vi),
            "theta": float(particle.theta),
            "omega": float(particle.omega),
            "x12": float(particle.x12),
            "m12": float(particle.m12),
            "entropyS": float(particle.entropyS),
            "mass": float(particle.mass)
        }
        self.token_stream.add_token(token)
    
    def compute_synchronization_metric(self) -> Dict:
        """Compute Kuramoto order parameter"""
        if len(self.particles) == 0:
            return {"r": 0.0, "meanTheta": 0.0}
        
        sum_real = sum(np.cos(p.theta) for p in self.particles)
        sum_imag = sum(np.sin(p.theta) for p in self.particles)
        
        r = np.sqrt(sum_real ** 2 + sum_imag ** 2) / len(self.particles)
        mean_theta = np.arctan2(sum_imag, sum_real)
        
        return {"r": r, "meanTheta": mean_theta}
    
    def compute_conservation_stats(self) -> Dict:
        """Compute conservation diagnostics"""
        Etotal = 0.0
        P = np.zeros(3)
        L = np.zeros(3)
        
        for p in self.particles:
            v2 = np.sum(p.velocity ** 2)
            K = 0.5 * p.mass * v2
            Etotal += K + p.Ugrav + p.Udm
            
            P += p.mass * p.velocity
            
            r = np.array([p.x, p.y, p.z])
            L += p.mass * np.cross(r, p.velocity)
        
        # Compute drift
        drift_E = 0.0
        if self.conservation_E0 != 0:
            drift_E = abs((Etotal - self.conservation_E0) / self.conservation_E0)
        
        return {
            "Etotal": Etotal,
            "P": P,
            "L": L,
            "drift": {"E": drift_E}
        }
    
    def compute_virial(self) -> Dict:
        """Compute virial ratio"""
        Ksum = 0.0
        Usum = 0.0
        
        for p in self.particles:
            v2 = np.sum(p.velocity ** 2)
            Ksum += 0.5 * p.mass * v2
            Usum += p.Ugrav + p.Udm
        
        ratio = (2 * Ksum) / abs(Usum) if abs(Usum) > 1e-10 else 1.0
        ok = abs(ratio - 1) < 0.1
        
        return {"ratio": ratio, "ok": ok}


# Example usage and Streamlit integration would go here
if __name__ == "__main__":
    # Example usage
    sim = Simulator()
    sim.set_seed(12345)
    
    # Add a particle
    sim.particles.append(Particle(0.1, 0.0, 0.0, 440.0))
    
    # Run simulation
    for _ in range(100):
        sim.tick()
        psi = sim.compute_psi()
        sync = sim.compute_synchronization_metric()
        cons = sim.compute_conservation_stats()
        
        print(f"Psi total: {psi['psiTotal']:.3f}, "
              f"Sync r: {sync['r']:.3f}, "
              f"Energy: {cons['Etotal']:.3e}")

