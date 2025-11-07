# -*- coding: utf-8 -*-
#!/usr/bin/env python3

##############################
# IMPORTS
##############################
import os
import sys
import json
import math
import time
import random
import logging
import threading
import queue
import glob
import pickle
import sqlite3
import traceback
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Audio
try:
    import sounddevice as sd
except ImportError:
    sd = None
    logging.warning("sounddevice is not installed. Audio processing will not be available.")

# Plotting / UI
try:
    import streamlit as st
except ImportError:
    st = None
    logging.warning("streamlit is not installed. Streamlit UI functionality will not be available.")

try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    plt.ion()  # Turn on interactive mode for real-time updates
except ImportError:
    plt = None
    Axes3D = None
    logging.warning("matplotlib is not installed. Plotting functionality will not be available.")

try:
    import plotly.graph_objects as go
except ImportError:
    go = None
    logging.warning("plotly is not installed. Interactive plotting functionality will not be available.")

try:
    import pandas as pd
except ImportError:
    pd = None
    logging.warning("pandas is not installed. Some chart functionality may be limited.")

# Performance acceleration and spatial tree
from numba import njit, prange
from scipy.spatial import cKDTree

# Extra modules for offline AI functionality
import tkinter as tk
from tkinter import scrolledtext
import pygame
try:
    import speech_recognition as sr
except ImportError:
    sr = None
    logging.warning("speech_recognition is not installed. Speech recognition functionality will not be available.")

try:
    import pyttsx3
except ImportError:
    pyttsx3 = None
    logging.warning("pyttsx3 is not installed. Text-to-speech functionality will not be available.")

try:
    from PIL import Image
except ImportError:
    Image = None
    logging.warning("PIL/Pillow is not installed. Image processing functionality will not be available.")

import collections

# Optional: Camera capture via OpenCV
try:
    import cv2
except ImportError:
    cv2 = None
    logging.warning("OpenCV is not installed. Camera functionality will not be available.")

# Optional: Mayavi for volumetric heatmaps
try:
    from mayavi import mlab  # pyright: ignore[reportMissingImports]
except ImportError:
    mlab = None
    logging.warning("Mayavi is not installed. Volumetric heatmap functionality will not be available.")

##############################
# GLOBAL CONSTANTS & LOGGING SETUP
##############################
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("CosmicSynapse")

# Simulation / Physics Constants
G = 6.67430e-11         # Gravitational constant (m^3 kg^-1 s^-2)
c = 3.0e8               # Speed of light (m/s)
h = 6.626e-34           # Planck's constant (J s)
k_B = 1.381e-23         # Boltzmann's constant (J/K)
E_0 = 1.0e3             # Reference energy (J)
alpha_initial = 1.0e-10   # Scaling factor
a_0 = 9.81              # Characteristic acceleration (m/s^2)
lambda_evo_initial = 1.0  # Evolution factor
E_replicate = 1e50      # Replication threshold (J)
MAX_PARTICLES = 1000    # Maximum number of particles
MAX_EXTRA_PARTICLES = 5000  # Maximum particles for extra mode visualization

DENSITY_PROFILE_PARAMS = {
    'rho0': 1e-24,        # Central density (kg/m^3)
    'rs': 1e21            # Scale radius (m)
}

# Audio settings
AUDIO_SAMPLERATE = 44100  
AUDIO_DURATION = 0.1     # seconds
FREQ_BINS = [500, 2000, 4000, 8000]

# Data directory (use raw string to avoid escape issues)
# Default to user's desktop/data if the original path doesn't exist
DATA_DIRECTORY = r"C:\Users\phera\Desktop\test\data"
if not os.path.exists(DATA_DIRECTORY):
    # Try to use current user's desktop
    user_home = os.path.expanduser("~")
    DATA_DIRECTORY = os.path.join(user_home, "Desktop", "data")
    if not os.path.exists(DATA_DIRECTORY):
        os.makedirs(DATA_DIRECTORY, exist_ok=True)

# Cosmic Brain File
COSMIC_BRAIN_FILE = "cosmic_brain.json"

# Ensure Streamlit session state variables
if st is not None:
    if "stop_threads" not in st.session_state:
        st.session_state["stop_threads"] = False
    if "plot_queue" not in st.session_state:
        st.session_state["plot_queue"] = queue.Queue()

# Global constants for extra mode (Pygame window)
WINDOW_WIDTH, WINDOW_HEIGHT = 800, 600

##############################
# MATH SYSTEM CLASS
##############################
class MathSystem:
    """
    Provides mathematical functions for evolving particle properties.
    """
    def __init__(self):
        self.parameters = {
            "base": 2,
            "offset": 1,
            "multiplier": 1,
            "chaos_factor": 0.5
        }
    
    def calculate(self, x):
        p = self.parameters
        chaos = math.sin(p["chaos_factor"] * x)
        return p["base"] ** (p["multiplier"] * x + p["offset"] + chaos)
    
    def evolve(self, dominant_freq, feedback=None):
        if feedback and "math_update" in feedback:
            self.parameters.update(feedback["math_update"])
        self.parameters["multiplier"] = 1 + (dominant_freq % 10) / 10
        self.parameters["offset"] = math.sin(dominant_freq / 100)
        self.parameters["base"] = max(1.1, 2 + (dominant_freq / 200))
        self.parameters["chaos_factor"] = 0.5 + math.cos(dominant_freq / 50)

math_system = MathSystem()

##############################
# NEURAL NETWORK DEFINITION
##############################
class ParticleNeuralNet(nn.Module):
    """
    Neural network for particle adaptation.
    """
    def __init__(self, input_size=14, hidden_size=64, output_size=2):
        super(ParticleNeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        return out

GLOBAL_MODEL = ParticleNeuralNet()
GLOBAL_OPTIMIZER = optim.SGD(GLOBAL_MODEL.parameters(), lr=0.001)

##############################
# FILE I/O FUNCTIONS
##############################
def scan_directory(directory_path: str) -> List[str]:
    """
    Scans the given directory recursively for JSON, DB, and BA2 files.
    """
    try:
        json_files = glob.glob(os.path.join(directory_path, '**', '*.json'), recursive=True)
        db_files = glob.glob(os.path.join(directory_path, '**', '*.db'), recursive=True)
        ba2_files = glob.glob(os.path.join(directory_path, '**', '*.ba2'), recursive=True)
        all_files = json_files + db_files + ba2_files
        logger.info(f"Found {len(json_files)} JSON, {len(db_files)} DB, and {len(ba2_files)} BA2 files in {directory_path}.")
        return all_files
    except Exception as e:
        logger.error(f"Error scanning directory {directory_path}: {e}")
        return []

def load_json_files(json_files: List[str]) -> List:
    """
    Loads and parses JSON files.
    """
    data = []
    for file in json_files:
        try:
            with open(file, 'r', encoding='utf-8', errors="replace") as f:
                data.append(json.load(f))
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error in {file}: {e}. Skipping this file.")
        except Exception as e:
            logger.error(f"Unexpected error loading {file}: {e}. Skipping this file.")
    logger.info(f"Loaded data from {len(data)} of {len(json_files)} JSON files.")
    return data

def load_db_files(db_files: List[str]) -> List:
    """
    Loads data from DB files by querying the 'data' table.
    """
    data = []
    for file in db_files:
        try:
            conn = sqlite3.connect(file)
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='data';")
            if cursor.fetchone():
                try:
                    cursor.execute("SELECT * FROM data;")
                    rows = cursor.fetchall()
                    data.extend(rows)
                    logger.info(f"Loaded {len(rows)} records from {file}.")
                except Exception as e:
                    logger.error(f"Error querying 'data' table in {file}: {e}. Skipping this file.")
            else:
                logger.warning(f"No 'data' table found in {file}. Skipping.")
            conn.close()
        except Exception as e:
            logger.error(f"Error loading DB file {file}: {e}")
    logger.info(f"Loaded data from {len(db_files)} DB files.")
    return data

def load_ba2_file(file_path: str):
    """
    Loads BA2 files as binary data.
    """
    try:
        with open(file_path, 'rb') as f:
            data = f.read()
        logger.info(f"Loaded BA2 file: {file_path}")
        return data
    except Exception as e:
        logger.error(f"Error loading BA2 file {file_path}: {e}")
        return None

def scan_data_for_all() -> List:
    """
    Scans the data directory and loads all relevant data.
    """
    all_files = scan_directory(DATA_DIRECTORY)
    json_data = load_json_files([f for f in all_files if f.endswith('.json')])
    db_data = load_db_files([f for f in all_files if f.endswith('.db')])
    ba2_data = [load_ba2_file(f) for f in all_files if f.endswith('.ba2') and load_ba2_file(f) is not None]
    return json_data + db_data + ba2_data

def load_simulation_data(filename: str) -> dict:
    """
    Loads simulation data from a pickle file.
    """
    try:
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        logger.info(f"Loaded simulation data from {filename}.")
        return data
    except Exception as e:
        logger.error(f"Error loading simulation data from {filename}: {e}")
        return {}

def save_cosmic_brain(frequencies: dict):
    """
    Saves frequency data to the cosmic_brain.json file.
    """
    try:
        if os.path.exists(COSMIC_BRAIN_FILE):
            with open(COSMIC_BRAIN_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            data = {}
        timestamp = time.time()
        data[str(timestamp)] = frequencies
        with open(COSMIC_BRAIN_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
        logger.info(f"Saved frequencies to {COSMIC_BRAIN_FILE} at timestamp {timestamp}.")
    except Exception as e:
        logger.error(f"Error saving to {COSMIC_BRAIN_FILE}: {e}")

##############################
# AUDIO CAPTURE FUNCTIONS
##############################
def audio_capture_thread(simulator, audio_queue, duration, sample_rate):
    """
    Captures audio data and places it in the audio queue.
    """
    if sd is None:
        logger.error("sounddevice is not available. Cannot capture audio.")
        return
    try:
        while simulator.audio_running:
            audio_data = sd.rec(int(sample_rate * duration), samplerate=sample_rate, channels=1, blocking=True)
            audio_queue.put(audio_data.flatten())
    except Exception as e:
        logger.error(f"Error in audio_capture_thread: {e}")

def process_audio(simulator, audio_queue, processed_audio_queue, sample_rate):
    """
    Processes audio data by performing FFT and extracting frequency magnitudes.
    """
    while simulator.audio_running or not audio_queue.empty():
        try:
            data = audio_queue.get()
            if data is None:
                break
            # Perform FFT
            fft_vals = np.fft.rfft(data)
            freqs = np.fft.rfftfreq(len(data), 1.0 / sample_rate)
            mags = np.abs(fft_vals)
            processed_audio_queue.put({'freqs': freqs.tolist(), 'mags': mags.tolist()})
        except Exception as e:
            logger.error(f"Error processing audio data: {e}")

##############################
# ADVANCED MATH FUNCTIONS (NUMBA)
##############################
@njit(parallel=True)
def compute_connectivity_numba(positions, masses, G, a0, N):
    """
    Computes the connectivity (Omega) between particles based on gravitational interactions.
    """
    Omega = np.zeros(N)
    for i in prange(N):
        for j in range(N):
            if i != j:
                dx = positions[j,0] - positions[i,0]
                dy = positions[j,1] - positions[i,1]
                dz = positions[j,2] - positions[i,2]
                r_sq = dx*dx + dy*dy + dz*dz + 1e-10  # Avoid division by zero
                r = math.sqrt(r_sq)
                Omega[i] += G * masses[j] / r_sq
    # Normalize Omega if needed
    Omega /= a0
    return Omega

@njit
def nfw_profile_density(r: float, rho0: float = 1e-24, rs: float = 1e21) -> float:
    """
    Computes the Navarro-Frenk-White (NFW) density profile.
    """
    EPSILON_DM = 1e-7
    if r < EPSILON_DM:
        r = EPSILON_DM
    return rho0 * (rs / r) / ((1.0 + (r / rs)) ** 2)

@njit
def advanced_dark_matter_potential(mass: float, x: float, y: float, z: float, G_val: float = 6.67430e-11, rho0: float = 1e-24, rs: float = 1e21) -> float:
    """
    Computes the dark matter potential based on the NFW profile.
    """
    r = math.sqrt(x*x + y*y + z*z)
    rho_val = nfw_profile_density(r, rho0, rs)
    return -G_val * mass * rho_val * (4.0 * math.pi * r * r)

@njit(parallel=True)
def update_particle_positions_numba(num_particles: int,
                                    masses: np.ndarray,
                                    positions: np.ndarray,
                                    velocities: np.ndarray,
                                    dt: float,
                                    G_val: float = 6.67430e-11):
    """
    Updates the positions and velocities of particles based on computed forces.
    """
    for i in prange(num_particles):
        fx = 0.0
        fy = 0.0
        fz = 0.0
        m1 = masses[i]
        x1, y1, z1 = positions[i, 0], positions[i, 1], positions[i, 2]
        for j in range(num_particles):
            if i == j:
                continue
            m2 = masses[j]
            x2, y2, z2 = positions[j, 0], positions[j, 1], positions[j, 2]
            dx = x2 - x1
            dy = y2 - y1
            dz = z2 - z1
            dist_sq = dx*dx + dy*dy + dz*dz + 1e-10
            dist = math.sqrt(dist_sq)
            f_mag = (G_val * m1 * m2) / dist_sq
            fx += f_mag * (dx / dist)
            fy += f_mag * (dy / dist)
            fz += f_mag * (dz / dist)
        ax = fx / m1
        ay = fy / m1
        az = fz / m1
        velocities[i, 0] += ax * dt
        velocities[i, 1] += ay * dt
        velocities[i, 2] += az * dt
    for i in prange(num_particles):
        positions[i, 0] += velocities[i, 0] * dt
        positions[i, 1] += velocities[i, 1] * dt
        positions[i, 2] += velocities[i, 2] * dt

##############################
# PARTICLE CLASS
##############################
class Particle:
    """
    Represents a cosmic particle with mass, position, velocity, energy, frequency, memory, entropy, and unique ID.
    """
    _id_counter = 0  # Class variable for unique IDs
    
    def __init__(self, mass, position, velocity, memory_size=10):
        self.id = Particle._id_counter
        Particle._id_counter += 1
        self.mass = mass
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.Ec = 0.5 * self.mass * np.linalg.norm(self.velocity)**2  # Kinetic Energy
        self.Uc = 0.0  # Potential Energy (to be updated)
        self.nu = (self.Ec + self.Uc) / h if h != 0 else 0.0
        self.memory = np.zeros(memory_size)
        self.S = self.compute_entropy()
        self.frequency = 0.0  # Assigned frequency
        self.tokens = []      # List to hold tokens
        
        # 12D CST Extensions: Internal adaptive state and synchronization
        self.x12 = 0.0        # Internal adaptive state (dimensionless, bounded [-1, 1])
        self.m12 = 0.0        # Memory of x12
        self.theta = random.uniform(0, 2 * math.pi)  # Phase variable for synchronization
        self.omega = 0.0      # Synaptic strength Ωi (will be computed by network)
        self.vi = (self.Ec + self.Uc) / h if h != 0 else 0.0  # Characteristic frequency (Ec/h)
    
    def update_position(self, force, dt):
        """
        Updates the particle's position and velocity based on the applied force.
        """
        try:
            acceleration = force / self.mass
            self.velocity += acceleration * dt
            self.position += self.velocity * dt
            logger.debug(f"Particle {self.id}: Position updated.")
        except Exception as e:
            logger.error(f"Error updating position for particle {self.id}: {e}")
    
    def update_energy(self, potential_energy):
        """
        Updates the particle's total energy by adding potential energy.
        """
        try:
            kinetic_energy = 0.5 * self.mass * np.linalg.norm(self.velocity)**2
            self.Ec = kinetic_energy if kinetic_energy > 0 else 0.0
            self.Uc = potential_energy
            self.nu = (self.Ec + self.Uc) / h if h != 0 else 0.0
            self.update_vi()  # Update characteristic frequency when energy changes
            self.update_entropy()
            logger.debug(f"Particle {self.id}: Energy updated. Kinetic: {self.Ec}, Potential: {self.Uc}")
        except Exception as e:
            logger.error(f"Error updating energy for particle {self.id}: {e}")
    
    def compute_entropy(self):
        """
        Computes the entropy of the particle based on its energy.
        """
        try:
            Ec_safe = self.Ec if self.Ec > 0 else E_0
            return k_B * math.log2(Ec_safe / E_0)
        except Exception as e:
            logger.error(f"Error computing entropy for particle {self.id}: {e}")
            return 0.0
    
    def update_entropy(self):
        """
        Updates the entropy of the particle.
        """
        try:
            self.S = self.compute_entropy()
            logger.debug(f"Particle {self.id}: Entropy updated.")
        except Exception as e:
            logger.error(f"Error updating entropy for particle {self.id}: {e}")
    
    def assign_frequency(self, frequency):
        """
        Assigns a frequency to the particle and generates a corresponding token.
        Updates particle energy based on frequency using E = h*nu relationship.
        """
        try:
            self.frequency = frequency
            # Update characteristic frequency
            self.nu = frequency
            # Update particle energy based on frequency (E = h*nu from quantum mechanics)
            # Add frequency-based energy contribution to existing energy
            if frequency > 0:
                frequency_energy = h * frequency
                # Add frequency energy to particle's cosmic energy
                self.Ec += frequency_energy * 0.1  # Scale factor to prevent excessive energy
                # Update entropy based on new energy
                self.update_entropy()
            token = f"Freq_{frequency:.2f}_Particle_{self.id}"
            self.tokens.append(token)
            logger.debug(f"Particle {self.id}: Assigned frequency {frequency:.2f} Hz, updated energy to {self.Ec:.2e} J")
        except Exception as e:
            logger.error(f"Error assigning frequency to particle {self.id}: {e}")
    
    def form_face_structure(self, target_position):
        """
        Adjusts the particle's position to form part of a face structure.
        """
        try:
            self.position = np.array(target_position)
            logger.debug(f"Particle {self.id}: Formed face structure at {self.position}.")
        except Exception as e:
            logger.error(f"Error forming face structure for particle {self.id}: {e}")
    
    def update_x12(self, dt, k=0.1, gamma=0.01):
        """
        Evolve internal adaptive state x12 according to: dx12/dt = k*omega - gamma*x12
        Steady-state: x12 = (k/gamma) * omega
        """
        try:
            dx12 = k * self.omega - gamma * self.x12
            self.x12 += dx12 * dt
            # Keep bounded to [-1, 1]
            self.x12 = max(-1.0, min(1.0, self.x12))
            logger.debug(f"Particle {self.id}: x12 updated to {self.x12:.4f}")
        except Exception as e:
            logger.error(f"Error updating x12 for particle {self.id}: {e}")
    
    def update_memory_state(self, dt, alpha=0.05):
        """
        Update memory m12 toward current x12 with time constant 1/alpha.
        dm12/dt = alpha * (x12 - m12)
        """
        try:
            dm12 = alpha * (self.x12 - self.m12)
            self.m12 += dm12 * dt
            logger.debug(f"Particle {self.id}: m12 updated to {self.m12:.4f}")
        except Exception as e:
            logger.error(f"Error updating memory state for particle {self.id}: {e}")
    
    def update_phase(self, dt, neighbors, Ksync=0.1):
        """
        Kuramoto-style phase evolution for synchronization.
        dtheta/dt = vi + (Ksync/N) * sum(sin(theta_j - theta_i))
        
        Args:
            dt: timestep
            neighbors: list of neighboring Particle objects
            Ksync: synchronization coupling strength
        """
        try:
            if not neighbors:
                # If no neighbors, just evolve with natural frequency
                self.theta += self.vi * dt
                return
            
            coupling_sum = 0.0
            for n in neighbors:
                coupling_sum += math.sin(n.theta - self.theta)
            
            dtheta = self.vi + (Ksync / len(neighbors)) * coupling_sum
            self.theta += dtheta * dt
            
            # Keep theta in [0, 2*pi)
            self.theta = self.theta % (2 * math.pi)
            logger.debug(f"Particle {self.id}: theta updated to {self.theta:.4f}")
        except Exception as e:
            logger.error(f"Error updating phase for particle {self.id}: {e}")
    
    def update_vi(self):
        """
        Update characteristic frequency vi = (Ec + Uc) / h
        """
        try:
            self.vi = (self.Ec + self.Uc) / h if h != 0 else 0.0
        except Exception as e:
            logger.error(f"Error updating vi for particle {self.id}: {e}")

##############################
# COSMIC NETWORK CLASS
##############################
class CosmicNetwork:
    """
    Calculates cosmic connectivity based on gravitational interactions.
    """
    def __init__(self, particles, a0=a_0):
        self.particles = particles
        self.a0 = a0
    
    def compute_connectivity(self, sigma=0.5, m0=1.0):
        """
        Computes the connectivity (Omega) for each particle with similarity weighting.
        
        The synaptic strength Ωi is computed as:
        omega_ij = (G * m_i * m_j) / (r_eff^2 * a0 * m0) * exp(-((x12_i - x12_j)^2) / (2 * sigma^2))
        Ωi = Σ ωij over neighbors
        
        Args:
            sigma: Gaussian similarity width parameter (default 0.5)
            m0: Reference mass for normalization (default 1.0)
        
        Returns:
            List of Omega values for each particle
        """
        try:
            positions = np.array([p.position for p in self.particles])
            masses = np.array([p.mass for p in self.particles])
            N = len(self.particles)
            
            # Compute base gravitational connectivity
            Omega_base = compute_connectivity_numba(positions, masses, G, self.a0, N)
            
            # Add Gaussian similarity weighting based on x12 differences
            for i, pi in enumerate(self.particles):
                omega_sum = 0.0
                for j, pj in enumerate(self.particles):
                    if i == j:
                        continue
                    
                    # Compute effective distance
                    r_ij = np.linalg.norm(positions[j] - positions[i])
                    r_eff = r_ij + 1e-10  # Avoid division by zero
                    
                    # Base gravitational coupling
                    omega_grav = (G * masses[i] * masses[j]) / (r_eff**2 * self.a0 * m0)
                    
                    # Similarity term: exp(-((x12_i - x12_j)^2) / (2 * sigma^2))
                    dx12 = pi.x12 - pj.x12
                    similarity = math.exp(-(dx12**2) / (2 * sigma**2))
                    
                    # Combined synaptic strength
                    omega_ij = omega_grav * similarity
                    omega_sum += omega_ij
                
                # Store total synaptic strength for particle i
                pi.omega = omega_sum
            
            logger.debug(f"Computed similarity-weighted connectivity for {N} particles.")
            return [p.omega for p in self.particles]
        except Exception as e:
            logger.error(f"Error computing connectivity: {e}")
            return np.zeros(len(self.particles))

##############################
# DYNAMICS CLASS
##############################
class Dynamics:
    """
    Handles dynamics computations.
    """
    def __init__(self, alpha=alpha_initial, lambda_evo_initial=lambda_evo_initial, gamma=0.01):
        self.alpha = alpha
        self.lambda_evo_initial = lambda_evo_initial
        self.gamma = gamma  # Damping factor
        
        # Path integral tracking for ψ normalization
        self.velocity_integral = 0.0  # ∫||v|| dt
        self.x12_integral = 0.0       # ∫Δx12 dt
        self.last_velocities = None
        self.last_x12_values = None
    
    def get_lambda_evo(self, t):
        """
        Computes the current lambda evolution factor based on time.
        """
        return self.lambda_evo_initial * math.exp(-self.gamma * t)
    
    def compute_Psi(self, Ec, Omega, positions, masses, particles=None, dt=0.0, phi=(1+math.sqrt(5))/2):
        """
        Computes Psi with full theoretical breakdown including all normalized terms.
        
        ψ breakdown includes:
        - φ·Ec/c² (golden ratio coupling)
        - λ (Lyapunov/cosmological constant)
        - ∫||v|| dt (path integral of velocity)
        - ∫Δx12 dt (internal state integral)
        - Ω·Ec (synaptic energy)
        - Ugrav + Udm (gravitational and dark matter potential)
        
        All terms are normalized with reference scales (m0, Eref, tref, vref).
        
        Args:
            Ec: array of cosmic energies
            Omega: array of synaptic strengths
            positions: array of particle positions
            masses: array of particle masses
            particles: list of Particle objects (optional, for x12 access)
            dt: timestep for path integrals (default 0.0)
            phi: golden ratio constant (default (1+√5)/2)
        
        Returns:
            If particles provided: dict with 'psi_total' array and 'breakdown' dict
            Otherwise: array of Psi values (backward compatible)
        """
        try:
            Ec = np.asarray(Ec)
            Omega = np.asarray(Omega)
            positions = np.asarray(positions)
            masses = np.asarray(masses)
            N = len(Ec)
            
            # Reference scales for normalization
            m0 = np.mean(masses) if len(masses) > 0 else 1.0
            Eref = E_0  # Reference energy
            tref = 1.0  # Reference time
            vref = c * 1e-6  # Reference velocity (small fraction of c)
            
            # Update path integrals if dt > 0 and particles provided
            if dt > 0 and particles is not None:
                velocities = np.array([np.linalg.norm(p.velocity) for p in particles])
                x12_values = np.array([p.x12 for p in particles])
                
                if self.last_velocities is not None:
                    # Trapezoidal integration
                    self.velocity_integral += dt * np.mean((velocities + self.last_velocities) / 2)
                    self.x12_integral += dt * np.mean(np.abs(x12_values - self.last_x12_values))
                
                self.last_velocities = velocities
                self.last_x12_values = x12_values
            
            # Initialize arrays for all terms
            term_phi = np.zeros(N)
            term_lambda = np.zeros(N)
            term_velocity = np.zeros(N)
            term_x12 = np.zeros(N)
            term_synaptic = np.zeros(N)
            term_grav = np.zeros(N)
            term_dm = np.zeros(N)
            psi_total = np.zeros(N)
            
            # Compute center of mass for reference
            com = np.mean(positions, axis=0) if len(positions) > 0 else np.zeros(3)
            r_com = np.linalg.norm(com) + 1e-10
            
            for i in range(N):
                # 1. φ·Ec/c² (golden ratio coupling, normalized)
                term_phi[i] = (phi * Ec[i]) / (c**2 * Eref)
                
                # 2. λ (Lyapunov/cosmological constant, normalized)
                term_lambda[i] = self.lambda_evo_initial / Eref
                
                # 3. ∫||v|| dt (path integral, normalized)
                if dt > 0:
                    v_norm = self.velocity_integral / (vref * tref)
                    term_velocity[i] = v_norm
                else:
                    # Placeholder: use current velocity magnitude
                    if particles is not None and i < len(particles):
                        v_mag = np.linalg.norm(particles[i].velocity)
                        term_velocity[i] = v_mag / vref
                
                # 4. ∫Δx12 dt (internal state integral, normalized)
                if dt > 0:
                    x12_norm = self.x12_integral / tref
                    term_x12[i] = x12_norm
                else:
                    # Placeholder: use current x12
                    if particles is not None and i < len(particles):
                        term_x12[i] = abs(particles[i].x12)
                
                # 5. Ω·Ec (synaptic energy, normalized)
                term_synaptic[i] = (Omega[i] * Ec[i]) / Eref
                
                # 6. Ugrav (gravitational potential, normalized)
                r_i = np.linalg.norm(positions[i] - com) + 1e-10
                U_grav = -G * masses[i] * np.sum(masses) / r_i
                term_grav[i] = U_grav / Eref
                
                # 7. Udm (dark matter potential, normalized)
                # Use advanced_dark_matter_potential function
                try:
                    pos = positions[i]
                    mass = masses[i]
                    U_dm = advanced_dark_matter_potential(mass, pos[0], pos[1], pos[2], G_val=G, rho0=1e-24, rs=1e21)
                    term_dm[i] = U_dm / Eref
                except Exception as e:
                    logger.debug(f"Error computing dark matter potential for particle {i}: {e}")
                    term_dm[i] = 0.0
            
            # Total Psi: sum of all normalized terms
            psi_total = (term_phi + term_lambda + term_velocity + 
                        term_x12 + term_synaptic + term_grav + term_dm)
            
            # If particles provided, return full breakdown
            if particles is not None:
                breakdown = {
                    "phi_term": term_phi,
                    "lambda_term": term_lambda,
                    "velocity_term": term_velocity,
                    "x12_term": term_x12,
                    "synaptic_term": term_synaptic,
                    "grav_term": term_grav,
                    "dm_term": term_dm,
                    "psi_total": psi_total
                }
                return breakdown
            else:
                # Backward compatible: return array
                return psi_total
                
        except Exception as e:
            logger.error(f"Error computing Psi: {e}")
            return np.zeros(len(Ec)) if isinstance(Ec, (list, np.ndarray)) else 0.0
    
    def compute_forces(self, Psi, positions):
        """
        Computes the forces acting on each particle.
        
        Args:
            Psi: Either array of Psi values or dict with 'psi_total' key
            positions: array of particle positions
        """
        try:
            # Handle both old format (array) and new format (dict)
            if isinstance(Psi, dict):
                Psi_values = Psi.get('psi_total', np.zeros(len(positions)))
            else:
                Psi_values = np.asarray(Psi)
            
            # Clamp Psi values to ±10 for stability
            Psi_values = np.clip(Psi_values, -10.0, 10.0)
            
            # Small modulation factor to prevent forces from becoming too large
            kappa = 1e-5
            
            N = len(Psi_values)
            forces = np.zeros((N, 3))
            for i in range(N):
                for j in range(N):
                    if i != j:
                        delta = positions[j] - positions[i]
                        distance_sq = np.dot(delta, delta) + 1e-10
                        distance = math.sqrt(distance_sq)
                        # Use Psi as modulation of gravity, not replacement
                        # Base gravitational force
                        base_force = G / distance_sq
                        # Modulate with Psi (scaled and bounded)
                        psi_modulation = 1.0 + kappa * Psi_values[i] * Psi_values[j]
                        force_magnitude = base_force * psi_modulation
                        forces[i] += force_magnitude * (delta / distance)
            return forces
        except Exception as e:
            logger.error(f"Error computing forces: {e}")
            return np.zeros((len(Psi_values) if isinstance(Psi, dict) else len(Psi), 3))

##############################
# 12D CST UTILITY FUNCTIONS
##############################

def compute_shannon_entropy(particles, bins=32):
    """
    Compute Shannon entropy over velocity distribution using histogram.
    S = -k_B * Σ(hist * log(hist + ε))
    
    This replaces the simple log2(Ec/E0) entropy with a proper
    statistical entropy over the velocity distribution.
    
    Args:
        particles: list of Particle objects
        bins: number of bins for velocity histogram (default 32)
    
    Returns:
        Shannon entropy value (J/K)
    """
    try:
        if not particles:
            return 0.0
        
        velocities = [np.linalg.norm(p.velocity) for p in particles]
        if not velocities or all(v == 0 for v in velocities):
            return 0.0
        
        # Create histogram
        hist, _ = np.histogram(velocities, bins=bins, density=True)
        
        # Avoid log(0) by adding small epsilon
        epsilon = 1e-12
        hist_safe = hist + epsilon
        
        # Compute Shannon entropy: S = -k_B * Σ(p * log(p))
        S = -k_B * np.sum(hist * np.log(hist_safe))
        
        # Update particle entropies
        for p in particles:
            p.S = S / len(particles)  # Distribute entropy equally (or use per-particle calculation)
        
        return S
    except Exception as e:
        logger.error(f"Error computing Shannon entropy: {e}")
        return 0.0


def compute_total_energy(particles):
    """
    Compute total energy of the system: E_total = Σ(Ec + Uc)
    
    Args:
        particles: list of Particle objects
    
    Returns:
        Total energy (J)
    """
    try:
        return sum(p.Ec + p.Uc for p in particles)
    except Exception as e:
        logger.error(f"Error computing total energy: {e}")
        return 0.0


def compute_total_momentum(particles):
    """
    Compute total momentum of the system: P_total = Σ(m * v)
    
    Args:
        particles: list of Particle objects
    
    Returns:
        Total momentum vector (kg·m/s)
    """
    try:
        if not particles:
            return np.zeros(3)
        return np.sum([p.mass * p.velocity for p in particles], axis=0)
    except Exception as e:
        logger.error(f"Error computing total momentum: {e}")
        return np.zeros(3)


def compute_total_angular_momentum(particles):
    """
    Compute total angular momentum: L_total = Σ(r × (m * v))
    
    Args:
        particles: list of Particle objects
    
    Returns:
        Total angular momentum vector (kg·m²/s)
    """
    try:
        if not particles:
            return np.zeros(3)
        return np.sum([np.cross(p.position, p.mass * p.velocity) for p in particles], axis=0)
    except Exception as e:
        logger.error(f"Error computing total angular momentum: {e}")
        return np.zeros(3)


def check_virial(particles):
    """
    Check virial theorem: 2<K> ≈ -<U> for bound systems.
    Returns the ratio: 2*K / (-U)
    For a virialized system, this should be close to 1.
    
    Args:
        particles: list of Particle objects
    
    Returns:
        Virial ratio (dimensionless)
    """
    try:
        if not particles:
            return 0.0
        
        # Total kinetic energy
        kinetic = sum(0.5 * p.mass * np.linalg.norm(p.velocity)**2 for p in particles)
        
        # Total potential energy
        potential = sum(p.Uc for p in particles)
        
        # Avoid division by zero
        if abs(potential) < 1e-10:
            return 0.0
        
        # Virial ratio
        ratio = (2 * kinetic) / (-potential)
        return ratio
    except Exception as e:
        logger.error(f"Error checking virial: {e}")
        return 0.0


def compute_adaptive_dt(particles, dt_max=1.0, safety_factor=0.1):
    """
    Compute adaptive timestep based on system dynamics.
    dt = min(dt_max, safety_factor * r_min / v_max)
    
    This ensures numerical stability by keeping the timestep
    small enough that particles don't move too far per step.
    
    Args:
        particles: list of Particle objects
        dt_max: maximum allowed timestep (default 1.0)
        safety_factor: safety factor for timestep (default 0.1)
    
    Returns:
        Adaptive timestep (s)
    """
    try:
        if not particles:
            return dt_max
        
        # Find maximum velocity
        v_max = max(np.linalg.norm(p.velocity) for p in particles)
        v_max = max(v_max, 1e-10)  # Avoid division by zero
        
        # Find minimum non-zero distance from origin
        distances = [np.linalg.norm(p.position) for p in particles if np.linalg.norm(p.position) > 0]
        if not distances:
            return dt_max
        
        r_min = min(distances)
        
        # Adaptive timestep
        dt_adaptive = safety_factor * r_min / v_max
        dt = min(dt_max, dt_adaptive)
        
        return dt
    except Exception as e:
        logger.error(f"Error computing adaptive timestep: {e}")
        return dt_max


def compute_synchronization_order_parameter(particles):
    """
    Compute Kuramoto synchronization order parameter r.
    r = |(1/N) * Σ(exp(i*theta))|
    
    r = 0: no synchronization (phases random)
    r = 1: perfect synchronization (all phases aligned)
    
    Args:
        particles: list of Particle objects
    
    Returns:
        Synchronization order parameter r (0 to 1)
    """
    try:
        if not particles:
            return 0.0
        
        N = len(particles)
        complex_sum = sum(np.exp(1j * p.theta) for p in particles)
        r = abs(complex_sum) / N
        
        return r
    except Exception as e:
        logger.error(f"Error computing synchronization order parameter: {e}")
        return 0.0


##############################
# REPLAY MANAGER CLASS
##############################
class ReplayManager:
    """
    Manages deterministic replay of simulation runs.
    Records audio frames and can replay them with fixed random seeds.
    """
    def __init__(self, seed=42):
        """
        Initialize replay manager with fixed random seed.
        
        Args:
            seed: random seed for reproducibility (default 42)
        """
        self.recorded_frames = []
        self.seed = seed
        self.is_recording = False
        self.is_replaying = False
        self.replay_index = 0
        
        # Set random seeds for reproducibility
        random.seed(self.seed)
        np.random.seed(self.seed)
        if torch is not None:
            torch.manual_seed(self.seed)
    
    def start_recording(self):
        """Start recording audio frames."""
        self.is_recording = True
        self.recorded_frames = []
        logger.info("ReplayManager: Started recording")
    
    def stop_recording(self):
        """Stop recording audio frames."""
        self.is_recording = False
        logger.info(f"ReplayManager: Stopped recording. Captured {len(self.recorded_frames)} frames")
    
    def record_frame(self, frame):
        """
        Record an audio frame (FFT results or similar).
        
        Args:
            frame: audio frame data (numpy array or similar)
        """
        if self.is_recording:
            self.recorded_frames.append(frame.copy() if hasattr(frame, 'copy') else frame)
    
    def start_replay(self):
        """Start replaying recorded frames."""
        if not self.recorded_frames:
            logger.warning("ReplayManager: No frames recorded, cannot replay")
            return False
        
        self.is_replaying = True
        self.replay_index = 0
        logger.info(f"ReplayManager: Started replaying {len(self.recorded_frames)} frames")
        return True
    
    def stop_replay(self):
        """Stop replaying."""
        self.is_replaying = False
        self.replay_index = 0
        logger.info("ReplayManager: Stopped replay")
    
    def get_next_frame(self):
        """
        Get next frame from replay sequence.
        
        Returns:
            Next frame or None if replay is complete
        """
        if not self.is_replaying:
            return None
        
        if self.replay_index >= len(self.recorded_frames):
            logger.info("ReplayManager: Replay complete")
            self.is_replaying = False
            return None
        
        frame = self.recorded_frames[self.replay_index]
        self.replay_index += 1
        return frame
    
    def reset_seed(self, seed=None):
        """
        Reset random seed for reproducibility.
        
        Args:
            seed: new seed (default: use existing seed)
        """
        if seed is not None:
            self.seed = seed
        random.seed(self.seed)
        np.random.seed(self.seed)
        if torch is not None:
            torch.manual_seed(self.seed)
        logger.info(f"ReplayManager: Reset seed to {self.seed}")


##############################
# DARK MATTER INFLUENCE CLASS
##############################
class DarkMatterInfluence:
    """
    Models dark matter's influence on particle energy.
    """
    def __init__(self, rho_d_func, params=DENSITY_PROFILE_PARAMS):
        self.rho_d_func = rho_d_func
        self.params = params
    
    def compute_delta_E_dark(self, particle_position, mass):
        """
        Computes the change in energy due to dark matter influence.
        """
        try:
            rho_d = self.rho_d_func(particle_position, self.params)
            distance = np.linalg.norm(particle_position) + 1e-10
            U_dark = -G * mass * rho_d / distance
            logger.debug(f"DarkMatterInfluence: U_dark computed for {particle_position}.")
            return U_dark
        except Exception as e:
            logger.error(f"Error computing dark matter influence: {e}")
            return 0.0

##############################
# REPLICATION CLASS
##############################
class Replication:
    """
    Handles particle replication.
    """
    def __init__(self, E_replicate=E_replicate):
        self.E_replicate = E_replicate
    
    def check_and_replicate(self, particles: List[Particle]) -> List[Particle]:
        """
        Checks particles for replication based on energy thresholds and replicates them if necessary.
        """
        new_particles = []
        try:
            for particle in particles:
                if particle.Ec > self.E_replicate and (len(particles) + len(new_particles) < MAX_PARTICLES):
                    new_mass = particle.mass * random.uniform(0.95, 1.05)
                    new_position = particle.position + np.random.normal(0, 1e9, 3)
                    new_velocity = particle.velocity * random.uniform(0.95, 1.05)
                    new_particle = Particle(new_mass, new_position.tolist(), new_velocity.tolist())
                    new_particle.Ec = particle.Ec * 0.5
                    new_particles.append(new_particle)
                    particle.Ec *= 0.5
                    logger.info(f"Particle {particle.id} replicated into {new_particle.id}.")
            return new_particles
        except Exception as e:
            logger.error(f"Error during replication: {e}")
            return new_particles

##############################
# LEARNING MECHANISM CLASS
##############################
class LearningMechanism:
    """
    Updates particle memory based on neighbor data and frequency features.
    """
    def __init__(self, memory_size=10, learning_rate=0.01):
        self.memory_size = memory_size
        self.learning_rate = learning_rate
    
    def update_memory(self, particle: Particle, neighbors: List[Particle], frequency_data=None):
        """
        Updates the memory of a particle based on its neighbors and frequency data.
        """
        try:
            average_Ec = np.mean([n.Ec for n in neighbors]) if neighbors else 0.0
            if frequency_data:
                freqs = frequency_data.get('freqs', [])
                mags = frequency_data.get('mags', [])
                if len(freqs) > 0 and len(mags) > 0 and len(freqs) == len(mags):
                    num_to_get = min(5, len(mags))
                    top_indices = np.argsort(mags)[-num_to_get:][::-1]
                    top_freqs = np.array(freqs)[top_indices]
                    top_mags = np.array(mags)[top_indices]
                    if np.max(top_mags) > 0:
                        top_mags = top_mags / np.max(top_mags)
                    # Pad or truncate to 5 elements
                    if len(top_mags) < 5:
                        frequency_features = np.pad(top_mags, (0, 5 - len(top_mags)), 'constant')
                    else:
                        frequency_features = top_mags[:5]
                else:
                    frequency_features = np.zeros(5)
            else:
                frequency_features = np.zeros(5)
            particle.memory = np.roll(particle.memory, -1)
            particle.memory[-5:] = frequency_features
            logger.debug(f"Memory updated for particle {particle.id}.")
        except Exception as e:
            logger.error(f"Error updating memory for particle {particle.id}: {e}")

##############################
# ADAPTIVE BEHAVIOR CLASS
##############################
class AdaptiveBehavior:
    """
    Enables particles to adapt their energy based on memory and neural network output.
    """
    def __init__(self, sensitivity=0.1):
        self.sensitivity = sensitivity
    
    def adapt(self, particle: Particle, neural_net, combined_input):
        """
        Adapts the particle's energy based on neural network output and combined input.
        """
        try:
            recent_input = combined_input.reshape(1, -1)
            neural_input = torch.tensor(recent_input, dtype=torch.float32)
            with torch.no_grad():
                output = neural_net(neural_input).numpy().flatten()
            delta_alpha = output[0]
            particle.Ec += self.sensitivity * delta_alpha * (recent_input.flatten()[-5:].mean() - particle.Ec)
            particle.Ec = max(particle.Ec, 0.0)
            particle.update_entropy()
            logger.debug(f"Particle {particle.id} energy adapted.")
        except Exception as e:
            logger.error(f"Error adapting behavior for particle {particle.id}: {e}")

##############################
# EXTRA FEATURES: FLOWER FORMATION & TEXT-TO-PARTICLE
##############################
def generate_flower_formation(simulator: 'Simulator'):
    """
    Rearranges particles into a flower-like formation.
    """
    num_petals = 6
    num_to_rearrange = min(100, len(simulator.particles))
    center = np.array([0.0, 0.0, 0.0])
    for i in range(num_to_rearrange):
        petal = i % num_petals
        radius = 1e10 + (i // num_petals) * 5e9
        angle = petal * (2 * math.pi / num_petals)
        x = center[0] + radius * math.cos(angle) + random.uniform(-1e9, 1e9)
        y = center[1] + radius * math.sin(angle) + random.uniform(-1e9, 1e9)
        z = center[2] + random.uniform(-1e9, 1e9)
        simulator.particles[i].position = np.array([x, y, z])
    logger.info("Particles rearranged into a flower formation.")

def generate_particles_from_text(simulator: 'Simulator', text: str):
    """
    Generates new particles based on the provided text input.
    """
    try:
        for char in text:
            freq = ord(char) * 10
            mass = np.random.uniform(1e20, 1e25)
            position = np.random.uniform(-1e11, 1e11, 3)
            velocity = np.random.uniform(-1e3, 1e3, 3)
            new_particle = Particle(mass, position, velocity)
            new_particle.nu = freq
            new_particle.Ec = 0.5 * mass * np.linalg.norm(velocity)**2
            new_particle.assign_frequency(freq)
            simulator.particles.append(new_particle)
            simulator.history[-1] = np.vstack([simulator.history[-1], new_particle.position])
        logger.info(f"Generated particles from text: {text}")
    except Exception as e:
        logger.error(f"Error generating particles from text: {e}")

##############################
# VISUAL PARTICLE CLASS (Extra Mode)
##############################
class VisualParticleExtra:
    """
    Represents a visual particle for extra (Pygame) mode.
    """
    def __init__(self, x, y, z, color, life, energy):
        self.x, self.y, self.z = x, y, z
        self.color = color
        self.life = life
        self.energy = energy
        self.radius = 1
    
    def update(self, dx, dy, dz, dominant_freq, target_position=None, motion_modifier=1.0):
        """
        Updates the particle's position and state.
        """
        freq_effect = math_system.calculate(self.energy)
        chaos = math.sin(dominant_freq / 50)
        if target_position:
            self.x += (target_position[0] - self.x) * 0.01
            self.y += (target_position[1] - self.y) * 0.01
            self.z += (target_position[2] - self.z) * 0.01
        else:
            self.x += dx * freq_effect * chaos * motion_modifier
            self.y += dy * freq_effect * motion_modifier
            self.z += dz * freq_effect * chaos * motion_modifier
        self.life -= 1
        self.radius += math.sin(self.energy) * 0.1
    
    def render(self, surface, zoom, camera_pos, rotation):
        """
        Renders the particle on the Pygame surface.
        """
        cx, cy, _ = camera_pos
        x = (self.x - cx) * zoom
        y = (self.y - cy) * zoom
        screen_x = int(WINDOW_WIDTH / 2 + x)
        screen_y = int(WINDOW_HEIGHT / 2 - y)
        pygame.draw.circle(surface, self.color, (screen_x, screen_y), max(1, int(self.radius)))

##############################
# REAL-TIME 3D VISUALIZER CLASS
##############################
class RealTime3DVisualizer:
    """
    Real-time 3D visualization using matplotlib that updates live like a game window.
    """
    def __init__(self, simulator: 'Simulator', update_interval=0.033):  # ~30 FPS
        self.simulator = simulator
        self.update_interval = update_interval
        self.fig = None
        self.ax = None
        self.scatter = None
        self.running = False
        self.last_update = time.time()
        self.animation_thread = None
        
    def initialize(self):
        """Initialize the 3D plot window."""
        if plt is None:
            logger.error("matplotlib is not available. Cannot create 3D visualization.")
            return False
        try:
            self.fig = plt.figure(figsize=(12, 10))
            self.ax = self.fig.add_subplot(111, projection='3d')
            self.ax.set_xlabel('X (m)')
            self.ax.set_ylabel('Y (m)')
            self.ax.set_zlabel('Z (m)')
            self.ax.set_title('Cosmic Synapse - Real-Time 3D Particle Visualization')
            self.ax.set_facecolor('black')
            self.fig.patch.set_facecolor('black')
            self.ax.xaxis.pane.fill = False
            self.ax.yaxis.pane.fill = False
            self.ax.zaxis.pane.fill = False
            self.ax.xaxis.pane.set_edgecolor('gray')
            self.ax.yaxis.pane.set_edgecolor('gray')
            self.ax.zaxis.pane.set_edgecolor('gray')
            self.ax.tick_params(colors='white')
            self.ax.xaxis.label.set_color('white')
            self.ax.yaxis.label.set_color('white')
            self.ax.zaxis.label.set_color('white')
            self.ax.title.set_color('white')
            self.running = True
            logger.info("Real-time 3D visualization initialized.")
            return True
        except Exception as e:
            logger.error(f"Error initializing 3D visualization: {e}")
            return False
    
    def update_plot(self, frequency_data=None):
        """Update the 3D plot with current particle positions."""
        if not self.running or self.fig is None or self.ax is None:
            return
        
        try:
            if len(self.simulator.particles) == 0 or len(self.simulator.history) == 0:
                return
            
            current_time = time.time()
            if current_time - self.last_update < self.update_interval:
                return
            
            # Get latest particle positions
            if len(self.simulator.history) > 0:
                positions = self.simulator.history[-1]
            else:
                positions = np.array([p.position for p in self.simulator.particles])
            
            if len(positions) == 0:
                return
            
            # Get particle properties
            Ec = np.array([p.Ec for p in self.simulator.particles])
            masses = np.array([p.mass for p in self.simulator.particles])
            frequencies = np.array([p.frequency for p in self.simulator.particles])
            
            # Normalize for visualization
            if np.max(Ec) > 0:
                norm_energy = Ec / np.max(Ec)
            else:
                norm_energy = np.zeros_like(Ec)
            
            if np.max(masses) > 0:
                sizes = 10 + (masses / np.max(masses)) * 100
            else:
                sizes = np.full(masses.shape, 10)
            
            # Determine colors based on frequency or energy
            if frequency_data and len(frequency_data.get('freqs', [])) > 0:
                freqs = frequency_data.get('freqs', [])
                mags = frequency_data.get('mags', [])
                if len(freqs) > 0 and len(mags) > 0:
                    dominant_freq = freqs[np.argmax(mags)]
                    hue = (dominant_freq / 20000.0) % 1.0
                    colors = plt.cm.hsv(norm_energy * hue)[:, :3]
                else:
                    colors = plt.cm.viridis(norm_energy)[:, :3]
            elif np.any(frequencies > 0):
                hue_values = (frequencies / 20000.0) % 1.0
                colors = plt.cm.hsv(hue_values)[:, :3]
            else:
                colors = plt.cm.viridis(norm_energy)[:, :3]
            
            # Clear previous scatter plot
            self.ax.clear()
            
            # Set axis properties again (they get cleared)
            self.ax.set_xlabel('X (m)')
            self.ax.set_ylabel('Y (m)')
            self.ax.set_zlabel('Z (m)')
            self.ax.set_title(f'Cosmic Synapse - Particles: {len(self.simulator.particles)} | Step: {len(self.simulator.history)-1}')
            self.ax.set_facecolor('black')
            self.ax.xaxis.pane.fill = False
            self.ax.yaxis.pane.fill = False
            self.ax.zaxis.pane.fill = False
            self.ax.xaxis.pane.set_edgecolor('gray')
            self.ax.yaxis.pane.set_edgecolor('gray')
            self.ax.zaxis.pane.set_edgecolor('gray')
            self.ax.tick_params(colors='white')
            self.ax.xaxis.label.set_color('white')
            self.ax.yaxis.label.set_color('white')
            self.ax.zaxis.label.set_color('white')
            self.ax.title.set_color('white')
            
            # Plot particles
            if len(positions) > 0:
                x = positions[:, 0]
                y = positions[:, 1]
                z = positions[:, 2]
                
                # Filter out invalid positions
                valid_mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
                if np.any(valid_mask):
                    x_valid = x[valid_mask]
                    y_valid = y[valid_mask]
                    z_valid = z[valid_mask]
                    colors_valid = colors[valid_mask]
                    sizes_valid = sizes[valid_mask]
                    
                    self.scatter = self.ax.scatter(x_valid, y_valid, z_valid, 
                                                   c=colors_valid, s=sizes_valid, 
                                                   alpha=0.7, edgecolors='none')
            
            # Set axis limits dynamically
            if len(positions) > 0 and np.any(np.isfinite(positions)):
                valid_positions = positions[np.isfinite(positions).all(axis=1)]
                if len(valid_positions) > 0:
                    margin = np.max(np.ptp(valid_positions, axis=0)) * 0.1
                    if margin > 0:
                        self.ax.set_xlim([np.min(valid_positions[:, 0]) - margin, 
                                         np.max(valid_positions[:, 0]) + margin])
                        self.ax.set_ylim([np.min(valid_positions[:, 1]) - margin, 
                                         np.max(valid_positions[:, 1]) + margin])
                        self.ax.set_zlim([np.min(valid_positions[:, 2]) - margin, 
                                         np.max(valid_positions[:, 2]) + margin])
            
            # Update the plot
            plt.draw()
            plt.pause(0.001)  # Small pause to allow GUI to update
            
            self.last_update = current_time
            
        except Exception as e:
            logger.error(f"Error updating 3D plot: {e}")
    
    def start_animation(self):
        """Start the animation loop in a separate thread."""
        if self.animation_thread is not None and self.animation_thread.is_alive():
            return
        
        def animation_loop():
            while self.running:
                try:
                    self.update_plot()
                    time.sleep(self.update_interval)
                except Exception as e:
                    logger.error(f"Error in animation loop: {e}")
                    time.sleep(0.1)
        
        self.animation_thread = threading.Thread(target=animation_loop, daemon=True)
        self.animation_thread.start()
        logger.info("3D animation thread started.")
    
    def stop(self):
        """Stop the visualization."""
        self.running = False
        if self.fig is not None:
            try:
                plt.close(self.fig)
            except:
                pass
        logger.info("3D visualization stopped.")

##############################
# VISUALIZER CLASS (Streamlit & Mayavi)
##############################
class Visualizer:
    """
    Visualizes the simulation using Plotly and Mayavi.
    """
    def __init__(self, simulator: 'Simulator'):
        self.simulator = simulator
    
    def plot_particles(self, step, frequency_data=None, plot_container=None):
        """
        Plots the positions of particles at a given simulation step.
        Returns the figure for real-time updates.
        """
        try:
            if step >= len(self.simulator.history) or len(self.simulator.history) == 0:
                return None
                
            positions = self.simulator.history[step]
            Ec = np.array([p.Ec for p in self.simulator.particles])
            masses = np.array([p.mass for p in self.simulator.particles])
            norm_energy = Ec / np.max(Ec) if np.max(Ec) > 0 else Ec
            sizes = 2 + (masses / np.max(masses)) * 8 if np.max(masses) > 0 else np.full(masses.shape, 2)
            
            # Determine colors based on frequency or energy
            if frequency_data:
                freqs = frequency_data.get('freqs', [])
                mags = frequency_data.get('mags', [])
                if len(freqs) > 0 and len(mags) > 0 and len(freqs) == len(mags):
                    dominant_freq = freqs[np.argmax(mags)]
                    hue = (dominant_freq / 20000) * 360
                    # Convert hue to RGB using matplotlib's hsv colormap
                    if plt:
                        color = plt.cm.hsv(hue / 360)
                        colors = np.tile(color[:3], (len(self.simulator.particles), 1))
                    else:
                        colors = norm_energy
                else:
                    colors = norm_energy
            else:
                colors = norm_energy
            
            # Use particle frequencies if available
            if len(self.simulator.particles) > 0:
                particle_freqs = np.array([p.frequency for p in self.simulator.particles])
                if np.any(particle_freqs > 0):
                    # Color by frequency
                    hue_values = (particle_freqs / 20000.0) % 1.0
                    if plt:
                        colors = np.array([plt.cm.hsv(h)[:3] for h in hue_values])
                    else:
                        colors = norm_energy
                else:
                    # Color by energy
                    colors = norm_energy
            
            # Create rotating camera angles for dynamic view
            angle = step * 0.01  # Slow rotation over time
            camera1 = dict(
                eye=dict(x=1.5 * np.cos(angle), y=1.5 * np.sin(angle), z=1.25)
            )
            camera2 = dict(
                eye=dict(x=-1.5 * np.cos(angle), y=-1.5 * np.sin(angle), z=1.25)
            )
            
            # Handle color format for Plotly
            if isinstance(colors, np.ndarray) and colors.ndim == 2:
                # Convert RGB to single value for colorscale
                color_values = norm_energy
            else:
                color_values = colors if isinstance(colors, np.ndarray) else norm_energy
            
            fig = go.Figure(data=[go.Scatter3d(
                x=positions[:,0],
                y=positions[:,1],
                z=positions[:,2],
                mode='markers',
                marker=dict(
                    size=sizes,
                    color=color_values,
                    colorscale='Viridis',
                    opacity=0.8,
                    colorbar=dict(title='Normalized Energy'),
                    line=dict(width=0)
                ),
                name=f'Step {step}'
            )])
            fig.update_layout(
                title=f"Real-Time 3D Particle Visualization - Step: {step} | Particles: {len(self.simulator.particles)}",
                scene=dict(
                    xaxis_title='X Position (m)',
                    yaxis_title='Y Position (m)',
                    zaxis_title='Z Position (m)',
                    bgcolor="black",
                    xaxis=dict(backgroundcolor="black", showbackground=True, gridcolor="gray"),
                    yaxis=dict(backgroundcolor="black", showbackground=True, gridcolor="gray"),
                    zaxis=dict(backgroundcolor="black", showbackground=True, gridcolor="gray"),
                    camera=camera1
                ),
                paper_bgcolor='black',
                font=dict(color='white'),
                height=600
            )
            
            if st is not None and plot_container is not None:
                # Update the container with new plot
                with plot_container:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.plotly_chart(fig, use_container_width=True, key=f"plot1_{step}")
                    with col2:
                        fig2 = fig
                        fig2.update_layout(scene_camera=camera2)
                        st.plotly_chart(fig2, use_container_width=True, key=f"plot2_{step}")
            elif st is not None:
                # Legacy mode - create new plots
                col1, col2 = st.columns(2)
                with col1:
                    st.plotly_chart(fig, use_container_width=True)
                with col2:
                    fig.update_layout(scene_camera=camera2)
                    st.plotly_chart(fig, use_container_width=True)
            
            return fig
        except Exception as e:
            logger.error(f"Error plotting particles at step {step}: {e}")
            return None
    
    def animate_simulation(self):
        """
        Creates an animation of the simulation over time.
        """
        try:
            frames = []
            for step in range(len(self.simulator.history)):
                positions = self.simulator.history[step]
                Ec = np.array([p.Ec for p in self.simulator.particles])
                norm_energy = Ec / np.max(Ec) if np.max(Ec) > 0 else Ec
                masses = np.array([p.mass for p in self.simulator.particles])
                sizes = 2 + (masses / np.max(masses)) * 8 if np.max(masses) > 0 else np.full(masses.shape, 2)
                freqs = []  # Placeholder for frequency data if needed
                mags = []
                if step < len(self.simulator.audio_history):
                    freqs = self.simulator.audio_history[step].get('freqs', [])
                    mags = self.simulator.audio_history[step].get('mags', [])
                if len(freqs) > 0 and len(mags) > 0 and len(freqs) == len(mags):
                    dominant_freq = freqs[np.argmax(mags)]
                    hue = (dominant_freq / 20000) * 360
                    color = plt.cm.hsv(hue / 360)
                    colors = np.tile(color[:3], (len(self.simulator.particles), 1))
                else:
                    colors = plt.cm.viridis(norm_energy)
                camera_angle = dict(eye=dict(x=1.25*math.cos(0.1*step), y=1.25*math.sin(0.1*step), z=1.25))
                frame = go.Frame(data=[go.Scatter3d(
                    x=positions[:,0],
                    y=positions[:,1],
                    z=positions[:,2],
                    mode='markers',
                    marker=dict(
                        size=sizes,
                        color=colors[:,0] if colors.ndim == 2 else colors,
                        colorscale='Viridis',
                        opacity=0.8,
                        showscale=False
                    )
                )],
                layout=go.Layout(scene_camera=camera_angle),
                name=str(step))
                frames.append(frame)
            fig = go.Figure(
                data=[go.Scatter3d(
                    x=self.simulator.history[0][:,0],
                    y=self.simulator.history[0][:,1],
                    z=self.simulator.history[0][:,2],
                    mode='markers',
                    marker=dict(
                        size=2,
                        color=np.zeros(len(self.simulator.particles)),
                        colorscale='Viridis',
                        opacity=0.8,
                        colorbar=dict(title='Normalized Energy')
                    )
                )],
                layout=go.Layout(
                    title="Cosmic Synapse Simulation Animation",
                    updatemenus=[dict(
                        type="buttons",
                        buttons=[dict(label="Play",
                                      method="animate",
                                      args=[None, {"frame": {"duration": 50, "redraw": True},
                                               "fromcurrent": True,
                                               "transition": {"duration": 0}}])])],
                    scene=dict(
                        xaxis_title='X Position (m)',
                        yaxis_title='Y Position (m)',
                        zaxis_title='Z Position (m)',
                        bgcolor="black",
                        xaxis=dict(backgroundcolor="black", showbackground=True),
                        yaxis=dict(backgroundcolor="black", showbackground=True),
                        zaxis=dict(backgroundcolor="black", showbackground=True),
                        camera=dict(eye=dict(x=1.25, y=1.25, z=1.25))
                    ),
                    paper_bgcolor='black',
                    font=dict(color='white')
                ),
                frames=frames
            )
            if st is not None:
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            logger.error(f"Error creating animation: {e}")
    
    def plot_volumetric_heatmap(self, step):
        """
        Plots a volumetric heatmap of particle energies using Mayavi.
        """
        try:
            if mlab is None:
                if st is not None:
                    st.error("Mayavi is not installed. Cannot render volumetric heatmap.")
                return
            positions = self.simulator.history[step]
            Ec = np.array([p.Ec for p in self.simulator.particles])
            grid_size = 100
            x = np.linspace(np.min(positions[:,0]), np.max(positions[:,0]), grid_size)
            y = np.linspace(np.min(positions[:,1]), np.max(positions[:,1]), grid_size)
            z = np.linspace(np.min(positions[:,2]), np.max(positions[:,2]), grid_size)
            X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
            density = np.zeros_like(X)
            for pos, energy in zip(positions, Ec):
                ix = np.searchsorted(x, pos[0]) - 1
                iy = np.searchsorted(y, pos[1]) - 1
                iz = np.searchsorted(z, pos[2]) - 1
                if 0 <= ix < grid_size and 0 <= iy < grid_size and 0 <= iz < grid_size:
                    density[ix, iy, iz] += energy
            density /= np.max(density) if np.max(density) > 0 else 1
            mlab.figure(bgcolor=(0, 0, 0))
            src = mlab.pipeline.scalar_field(X, Y, Z, density)
            mlab.pipeline.volume(src, vmin=0.1, vmax=1.0, opacity='auto')
            mlab.title(f"Volumetric Heatmap - Step {step}", color=(1,1,1), size=0.5)
            mlab.show()
        except Exception as e:
            logger.error(f"Error creating volumetric heatmap at step {step}: {e}")
    
    def plot_metrics(self):
        """
        Plots various simulation metrics over time.
        """
        try:
            steps_arr = np.arange(1, len(self.simulator.total_energy_history) + 1) * self.simulator.dt
            fig, axs = plt.subplots(2, 2, figsize=(15, 10))
            axs[0,0].plot(steps_arr, self.simulator.total_energy_history, label='Total Energy', color='blue')
            axs[0,0].set_xlabel('Time (s)')
            axs[0,0].set_ylabel('Energy (J)')
            axs[0,0].set_title('Total Energy Over Time')
            axs[0,0].legend()
            axs[0,0].grid(True)
            axs[0,1].plot(steps_arr, self.simulator.kinetic_energy_history, label='Kinetic Energy', color='green')
            axs[0,1].plot(steps_arr, self.simulator.potential_energy_history, label='Potential Energy', color='red')
            axs[0,1].set_xlabel('Time (s)')
            axs[0,1].set_ylabel('Energy (J)')
            axs[0,1].set_title('Kinetic and Potential Energy Over Time')
            axs[0,1].legend()
            axs[0,1].grid(True)
            axs[1,0].plot(steps_arr, self.simulator.entropy_history, label='Average Entropy', color='purple')
            axs[1,0].set_xlabel('Time (s)')
            axs[1,0].set_ylabel('Entropy (J)')
            axs[1,0].set_title('Average Entropy Over Time')
            axs[1,0].legend()
            axs[1,0].grid(True)
            particle_counts = [len(p) for p in self.simulator.history]
            axs[1,1].plot(steps_arr, particle_counts, label='Number of Particles', color='orange')
            axs[1,1].set_xlabel('Time (s)')
            axs[1,1].set_ylabel('Count')
            axs[1,1].set_title('Number of Particles Over Time')
            axs[1,1].legend()
            axs[1,1].grid(True)
            plt.tight_layout()
            if st is not None:
                st.pyplot(fig)
        except Exception as e:
            logger.error(f"Error plotting metrics: {e}")

##############################
# SIMULATOR CLASS (Main Simulation)
##############################
class Simulator:
    """
    Orchestrates the simulation, integrating all components.
    """
    """
    Notes:
    - Particles will react to audio frequencies and camera brightness.
    - Specific particles will form a mouth structure based on audio intensity.
    - The Tkinter logs window will display audio frequencies and camera brightness data.
    """
    def __init__(self, num_particles=100, steps=1000, dt=1.0, data_directory=DATA_DIRECTORY):
        self.particles = self.initialize_particles(num_particles)
        self.network = CosmicNetwork(self.particles)
        self.dynamics = Dynamics()
        self.dark_matter = DarkMatterInfluence(self.density_function)
        self.replication = Replication()
        self.learning = LearningMechanism()
        self.adaptive = AdaptiveBehavior()
        self.visualizer = Visualizer(self)
        
        # Real-time 3D visualizer for live updates (only for non-Streamlit modes)
        self.realtime_3d_visualizer = None
        # Only initialize matplotlib window if not in Streamlit mode
        if plt is not None and st is None:
            try:
                self.realtime_3d_visualizer = RealTime3DVisualizer(self, update_interval=0.033)
                if self.realtime_3d_visualizer.initialize():
                    self.realtime_3d_visualizer.start_animation()
                    logger.info("Real-time 3D visualization started.")
            except Exception as e:
                logger.warning(f"Could not initialize real-time 3D visualization: {e}")
                self.realtime_3d_visualizer = None
        
        self.steps = steps
        self.dt = dt
        self.dt0 = dt  # Store original timestep for adaptive recovery
        self.time = 0
        self.step = 0  # Step counter
        self.history = [np.array([p.position for p in self.particles])]
        self.total_energy_history = []
        self.kinetic_energy_history = []
        self.potential_energy_history = []
        self.entropy_history = []
        self.octree = None
        self.octree_size = 0
        self.audio_queue = queue.Queue()
        self.processed_audio_queue = queue.Queue()
        self.audio_thread = None
        self.processing_thread = None
        self.audio_running = False
        # Camera integration variables
        self.camera_running = False
        self.camera_queue = queue.Queue()
        self.processed_camera_queue = queue.Queue()
        self.camera_thread = None
        self._last_camera_frame = None  # Initialize camera frame tracking
        self.data_directory = data_directory
        # Data scan cache to avoid heavy directory scans every step
        self._data_scan_cache = None
        self._data_scan_cache_time = 0
        self._data_scan_cache_interval = 10.0  # Refresh every 10 seconds
        # Cosmic Brain Frequency Data
        self.cosmic_brain = self.load_cosmic_brain()
        # Audio History for Visualization
        self.audio_history = []
        # 12D CST: Psi breakdown storage for diagnostics
        self.psi_breakdown = None
        # 12D CST: Replay manager for deterministic runs
        self.replay_manager = ReplayManager(seed=42)
        logger.info(f"Initialized {num_particles} particles.")
    
    def initialize_particles(self, N) -> List[Particle]:
        """
        Initializes particles with random masses, positions, and velocities.
        """
        try:
            masses = np.random.uniform(1e20, 1e25, N)
            positions = np.random.uniform(-1e11, 1e11, (N, 3))
            velocities = np.random.uniform(-1e3, 1e3, (N, 3))
            particles = [Particle(m, pos, vel) for m, pos, vel in zip(masses, positions, velocities)]
            return particles
        except Exception as e:
            logger.error(f"Error initializing particles: {e}")
            return []
    
    def density_function(self, position, params) -> float:
        """
        Computes the dark matter density at a given position using the NFW profile.
        """
        try:
            rho0 = params.get('rho0', 1e-24)
            rs = params.get('rs', 1e21)
            r = np.linalg.norm(position) + 1e-10
            return rho0 / ((r/rs) * (1 + r/rs)**2)
        except Exception as e:
            logger.error(f"Error in density function: {e}")
            return 0.0
    
    def compute_total_energy(self) -> float:
        """
        Computes the total energy of the system, including kinetic and potential energies.
        """
        try:
            kinetic_energy = np.sum([0.5 * p.mass * np.linalg.norm(p.velocity)**2 for p in self.particles])
            potential_energy = 0.0
            N = len(self.particles)
            positions = np.array([p.position for p in self.particles])
            masses = np.array([p.mass for p in self.particles])
            for i in range(N):
                r_i = positions[i]
                delta = positions[i+1:] - r_i
                distances = np.linalg.norm(delta, axis=1) + 1e-10
                potential_energy -= np.sum(G * masses[i] * masses[i+1:] / distances)
            total_Ec = np.sum([p.Ec for p in self.particles])
            total_energy = kinetic_energy + potential_energy + total_Ec
            self.total_energy_history.append(total_energy)
            self.kinetic_energy_history.append(kinetic_energy)
            self.potential_energy_history.append(potential_energy)
            avg_entropy = np.mean([p.S for p in self.particles])
            self.entropy_history.append(avg_entropy)
            logger.debug(f"Computed total energy: {total_energy} J")
            return total_energy
        except Exception as e:
            logger.error(f"Error computing total energy: {e}")
            return 0.0
    
    def run_step(self, neural_net, frequency_data=None, camera_data=None):
        """
        Executes a single simulation step, updating particle states and processing interactions.
        
        AUDIO PROCESSING FLOW (Core Engine Integration):
        Each sound/blow captured by microphone is processed through the core engine:
        1. Audio -> FFT -> frequency_data (freqs, mags) passed to this method
        2. Frequencies assigned to particles via assign_frequency() -> updates particle.Ec, particle.nu
        3. Updated Ec affects Psi computation (informational energy density)
        4. Psi affects force computation -> forces affect particle positions
        5. Frequency data updates particle memory -> neural network adaptation -> Ec changes
        6. All particle reactions come through core engine methods (update_position, update_energy, etc.)
        """
        try:
            # Scan directory for data files
            all_files = scan_directory(self.data_directory)
            json_files = [f for f in all_files if f.endswith('.json')]
            db_files = [f for f in all_files if f.endswith('.db')]
            json_data = load_json_files(json_files)
            db_data = load_db_files(db_files)
            combined_data = json_data + db_data  # (For potential future use)
            
            # Process frequency data
            if frequency_data:
                freqs = frequency_data.get('freqs', [])
                mags = frequency_data.get('mags', [])
                if len(freqs) > 0 and len(mags) > 0 and len(freqs) == len(mags):
                    # Ensure we don't try to get more indices than exist
                    num_to_get = min(len(FREQ_BINS), len(mags))
                    top_indices = np.argsort(mags)[-num_to_get:][::-1]
                    top_freqs = np.array(freqs)[top_indices]
                    top_mags = np.array(mags)[top_indices]
                    # Pad with zeros if not enough frequencies
                    if len(top_freqs) < len(FREQ_BINS):
                        pad_length = len(FREQ_BINS) - len(top_freqs)
                        top_freqs = np.pad(top_freqs, (0, pad_length), 'constant')
                        top_mags = np.pad(top_mags, (0, pad_length), 'constant')
                    else:
                        top_freqs = top_freqs[:len(FREQ_BINS)]
                        top_mags = top_mags[:len(FREQ_BINS)]
                else:
                    top_freqs = np.zeros(len(FREQ_BINS))
                    top_mags = np.zeros(len(FREQ_BINS))
                # Normalize magnitudes
                if np.max(top_mags) > 0:
                    top_mags = top_mags / np.max(top_mags)
                else:
                    top_mags = top_mags
                # Save frequencies to cosmic_brain
                frequencies = {"freqs": top_freqs.tolist(), "mags": top_mags.tolist()}
                save_cosmic_brain(frequencies)
                self.audio_history.append(frequencies)
            else:
                top_freqs = np.zeros(len(FREQ_BINS))
                top_mags = np.zeros(len(FREQ_BINS))
            
            # Store top_mags for use in continuous learning updates
            freq_features_for_learning = top_mags.copy() if len(top_mags) > 0 else np.zeros(4)
            if len(freq_features_for_learning) < 4:
                freq_features_for_learning = np.pad(freq_features_for_learning, (0, 4 - len(freq_features_for_learning)), 'constant')
            elif len(freq_features_for_learning) > 4:
                freq_features_for_learning = freq_features_for_learning[:4]
            
            # Assign frequencies to particles - CORE ENGINE PROCESSING
            # Each sound/blow captured by microphone is processed here and affects particles through core methods
            frequency_particle_map = {}
            # Ensure top_freqs is iterable (convert numpy array to list if needed)
            if isinstance(top_freqs, np.ndarray):
                top_freqs_list = top_freqs.tolist()
            else:
                top_freqs_list = list(top_freqs) if top_freqs else []
            for i, freq in enumerate(top_freqs_list):
                if i < len(self.particles):
                    particle = self.particles[i]
                    # Assign frequency to particle - this updates particle.nu, particle.Ec, and particle.entropy
                    # Using core engine method: assign_frequency() -> updates Ec -> affects Psi -> affects forces
                    particle.assign_frequency(float(freq))
                    frequency_particle_map[float(freq)] = particle
                else:
                    break  # No more particles to assign
            
            # Generate tokens based on frequencies
            tokens = [f"Freq_{freq:.2f}_Particle_{frequency_particle_map[float(freq)].id}" for freq in top_freqs_list if float(freq) > 0 and float(freq) in frequency_particle_map]
            
            # Prepare NN input: memory (10) + top_mags (4)
            # Adjusted input size based on memory_size (10) + len(FREQ_BINS) (4)
            input_vectors = [np.concatenate((p.memory, top_mags)) for p in self.particles]
            neural_input = np.array(input_vectors)
            neural_input_tensor = torch.tensor(neural_input, dtype=torch.float32)
            
            # Compute connectivity with similarity weighting (uses particle positions, masses, and x12 states)
            Omega = self.network.compute_connectivity(sigma=0.5, m0=1.0)
            Ec = np.array([p.Ec for p in self.particles])
            positions = np.array([p.position for p in self.particles])
            masses = np.array([p.mass for p in self.particles])
            
            # Compute Psi (informational energy density - includes frequency contributions through Ec)
            # Audio frequencies affect Ec, which affects Psi, which affects forces
            # Pass particles and dt for full 12D CST breakdown
            Psi_result = self.dynamics.compute_Psi(Ec, Omega, positions, masses, 
                                                   particles=self.particles, dt=self.dt)
            
            # Handle both old format (array) and new format (dict with breakdown)
            if isinstance(Psi_result, dict):
                Psi = Psi_result['psi_total']
                # Store breakdown for diagnostics if needed
                self.psi_breakdown = Psi_result
            else:
                Psi = Psi_result
            
            # Compute dark matter influence
            delta_E_dark = np.array([self.dark_matter.compute_delta_E_dark(p.position, p.mass) for p in self.particles])
            for i, particle in enumerate(self.particles):
                particle.Ec += delta_E_dark[i]
                if particle.Ec < 0:
                    particle.Ec = 0.0
            
            # Get current lambda value
            current_lambda = self.dynamics.get_lambda_evo(self.time)
            self.dynamics.alpha = current_lambda  # Example usage
            
            # Compute forces (Psi contains frequency contributions through Ec -> forces affected by audio)
            forces = self.dynamics.compute_forces(Psi, positions)
            
            # Update particle positions and energies using core engine methods
            # Forces computed from Psi (which includes frequency contributions through Ec)
            # Audio frequencies directly affect particle movement through: audio -> Ec -> Psi -> forces -> position
            for i, particle in enumerate(self.particles):
                # Update position using forces (affected by audio through Psi -> forces chain)
                particle.update_position(forces[i], self.dt)
                # Calculate potential energy (gravitational interactions)
                potential_energy = -G * np.sum([
                    other_p.mass / (np.linalg.norm(particle.position - other_p.position) + 1e-10)
                    for j, other_p in enumerate(self.particles) if j != i
                ])
                # Update energy (this also updates nu = (Ec + Uc)/h, which reflects audio frequency)
                particle.update_energy(potential_energy)
            
            # 12D CST Extensions: Update internal adaptive states and synchronization
            # Update x12 (internal adaptive state) for each particle
            for particle in self.particles:
                particle.update_x12(self.dt, k=0.1, gamma=0.01)
                particle.update_memory_state(self.dt, alpha=0.05)
            
            # Update phase synchronization (Kuramoto model)
            Ksync = 0.1  # Synchronization coupling strength
            for particle in self.particles:
                neighbors = self.find_neighbors(particle)
                particle.update_phase(self.dt, neighbors, Ksync=Ksync)
            
            # Update entropy using Shannon entropy (replaces simple log2 calculation)
            compute_shannon_entropy(self.particles, bins=32)
            
            # Conservation diagnostics
            total_energy = compute_total_energy(self.particles)
            total_momentum = compute_total_momentum(self.particles)
            total_angular_momentum = compute_total_angular_momentum(self.particles)
            virial_ratio = check_virial(self.particles)
            sync_order_param = compute_synchronization_order_parameter(self.particles)
            
            # Log conservation diagnostics (optional, can be controlled by logging level)
            if self.step % 100 == 0:  # Log every 100 steps
                logger.debug(f"Step {self.step}: E_total={total_energy:.2e}, "
                           f"Virial={virial_ratio:.3f}, Sync_r={sync_order_param:.3f}")
            
            # Adaptive timestep - dynamically adjust for numerical stability with recovery
            adaptive_dt = compute_adaptive_dt(self.particles, dt_max=self.dt0, safety_factor=0.1)
            # Clamp dt between 10% of original and original value, allowing recovery
            self.dt = min(self.dt0, max(adaptive_dt, 0.1 * self.dt0))
            
            # Increment step counter and time
            self.step += 1
            self.time += self.dt
    
            # Learning and adaptation - audio frequencies processed through core engine mechanisms
            # Audio frequencies affect particle memory and adaptation through core engine methods
            for particle in self.particles:
                neighbors = self.find_neighbors(particle)
                # Update memory with frequency data (core learning mechanism - processes audio input)
                self.learning.update_memory(particle, neighbors, frequency_data=frequency_data)
                # Adaptive behavior uses neural network with frequency-influenced memory
                combined_input = np.concatenate((particle.memory, top_mags))
                # This adapts particle energy based on frequency data through core adaptive engine
                # Each sound/blow is processed: frequency -> memory -> neural network -> Ec adaptation
                self.adaptive.adapt(particle, neural_net, combined_input)
            
            # Replication and data token processing...
            new_particles = self.replication.check_and_replicate(self.particles)
            self.particles.extend(new_particles)
            
            # Cross-reference frequency data to generate particles from tokens
            if tokens:
                for token in tokens:
                    word = self.map_token_to_word(token)
                    if word:
                        data_tokens = self.scan_data_for_word(word)
                        for data_token in data_tokens:
                            self.generate_particles_from_token(data_token)
                        # Form specific structures based on word
                        if word.lower() == "flower":
                            generate_flower_formation(self)
                        generate_particles_from_text(self, word)
                        if word.lower() in ["hello", "world", "face", "mouth"]:
                            self.form_face()
            
            # Handle camera data for real-time particle formation
            if camera_data:
                self.process_camera_data(camera_data)
            
            # Update history and compute total energy
            self.history.append(np.array([p.position for p in self.particles]))
            self.compute_total_energy()
            
            # Continuous learning update - every step ensures constant learning
            # This ensures the system is constantly learning and adapting
            # Update all particles to ensure comprehensive learning
            try:
                for particle in self.particles:
                    neighbors = self.find_neighbors(particle)
                    self.learning.update_memory(particle, neighbors, frequency_data)
                    # Also update adaptive behavior for continuous learning
                    if len(neighbors) > 0:
                        # Prepare input for neural network (memory + frequency features)
                        combined_input = np.concatenate((particle.memory, freq_features_for_learning))
                        if len(combined_input) >= 14:  # Ensure we have enough input
                            combined_input = combined_input[:14]
                            self.adaptive.adapt(particle, GLOBAL_MODEL, combined_input)
            except Exception as e:
                logger.debug(f"Continuous learning update skipped: {e}")
            
            # Update real-time 3D visualization every step
            if self.realtime_3d_visualizer is not None and self.realtime_3d_visualizer.running:
                try:
                    self.realtime_3d_visualizer.update_plot(frequency_data)
                except Exception as e:
                    logger.debug(f"3D visualization update skipped: {e}")
            
            self.time += self.dt
            logger.info(f"Completed step {self.time} with {len(self.particles)} particles.")
        except Exception as e:
            logger.error(f"Error during simulation step: {e}")
    
    def find_neighbors(self, particle, radius=1e10) -> List[Particle]:
        """
        Finds neighboring particles within a specified radius using a spatial tree.
        """
        try:
            if not self.octree or self.octree_size != len(self.particles):
                positions = np.array([p.position for p in self.particles])
                self.octree = cKDTree(positions)
                self.octree_size = len(self.particles)
            indices = self.octree.query_ball_point(particle.position, r=radius)
            return [self.particles[i] for i in indices if self.particles[i] != particle]
        except Exception as e:
            logger.error(f"Error finding neighbors for particle {particle.id}: {e}")
            return []
    
    def map_token_to_word(self, token: str) -> str:
        """
        Maps a token to a word based on predefined rules.
        """
        # Example mapping based on frequency or token content
        if "flower" in token.lower():
            return "flower"
        elif "hello" in token.lower():
            return "hello"
        elif "world" in token.lower():
            return "world"
        elif "face" in token.lower():
            return "face"
        elif "mouth" in token.lower():
            return "mouth"
        else:
            return "unknown"
    
    def scan_data_for_word(self, word: str) -> List:
        """
        Scans data files for the given word and returns relevant tokens.
        """
        data_tokens = []
        try:
            all_files = scan_directory(self.data_directory)
            json_files = [f for f in all_files if f.endswith('.json')]
            db_files = [f for f in all_files if f.endswith('.db')]
            for file in json_files:
                try:
                    with open(file, 'r', encoding='utf-8', errors="replace") as f:
                        content = f.read()
                        if word.lower() in content.lower():
                            data_tokens.append({'file': file, 'content': content})
                except Exception as e:
                    logger.error(f"Error scanning JSON file {file}: {e}")
            for file in db_files:
                try:
                    conn = sqlite3.connect(file)
                    cursor = conn.cursor()
                    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='data';")
                    if cursor.fetchone():
                        cursor.execute("SELECT * FROM data;")
                        rows = cursor.fetchall()
                        for row in rows:
                            for item in row:
                                if isinstance(item, str) and word.lower() in item.lower():
                                    data_tokens.append({'file': file, 'content': item})
                    conn.close()
                except Exception as e:
                    logger.error(f"Error scanning DB file {file}: {e}")
            logger.info(f"Found {len(data_tokens)} tokens for word '{word}'.")
        except Exception as e:
            logger.error(f"Error scanning data for word '{word}': {e}")
        return data_tokens
    
    def generate_particles_from_token(self, token):
        """
        Generates new particles based on the content of a token.
        """
        try:
            content = token.get('content', '') if isinstance(token, dict) else str(token)
            for char in content:
                freq = ord(char) * 10
                mass = np.random.uniform(1e20, 1e25)
                pos = np.random.uniform(-1e11, 1e11, 3)
                vel = np.random.uniform(-1e3, 1e3, 3)
                new_particle = Particle(mass, pos, vel)
                new_particle.nu = freq
                new_particle.Ec = 0.5 * mass * np.linalg.norm(vel)**2
                new_particle.assign_frequency(freq)
                self.particles.append(new_particle)
                self.history[-1] = np.vstack([self.history[-1], new_particle.position])
            logger.info("Generated new particle(s) from token.")
        except Exception as e:
            logger.error(f"Error generating particle from token: {e}")
    
    def form_face(self):
        """
        Forms a facial structure with particles based on learned frequencies.
        """
        try:
            # Simple face structure: eyes, nose, mouth
            face_positions = [
                # Eyes
                (-1e10, 2e10, 0),
                (1e10, 2e10, 0),
                # Nose
                (0, 0, 0),
                # Mouth corners
                (-1e10, -2e10, 0),
                (1e10, -2e10, 0)
            ]
            for i, pos in enumerate(face_positions):
                if i < len(self.particles):
                    self.particles[i].form_face_structure(pos)
            logger.info("Particles arranged to form a face structure.")
        except Exception as e:
            logger.error(f"Error forming face structure: {e}")
    
    def process_camera_data(self, camera_data):
        """
        Processes camera data to influence particle behaviors.
        """
        try:
            # Example: Use average brightness to adjust particle colors or positions
            avg_brightness = camera_data.get('brightness', 0.0)
            # Map brightness to some parameter, e.g., movement speed
            movement_speed = avg_brightness * 0.1
            # Influence specific particles (e.g., mouth particles) based on brightness
            mouth_particle_indices = [3, 4]  # Assuming these indices correspond to mouth corners
            for idx in mouth_particle_indices:
                if idx < len(self.particles):
                    particle = self.particles[idx]
                    # Adjust velocity based on brightness
                    particle.velocity += np.random.uniform(-movement_speed, movement_speed, 3)
                    # Optionally, adjust position to simulate opening/closing mouth
                    particle.position[1] += movement_speed * 1e9  # Example adjustment
            logger.info(f"Processed camera data: Avg Brightness = {avg_brightness}")
        except Exception as e:
            logger.error(f"Error processing camera data: {e}")
    
    def load_cosmic_brain(self) -> dict:
        """
        Loads the cosmic_brain.json file containing frequency data.
        """
        try:
            if os.path.exists(COSMIC_BRAIN_FILE):
                with open(COSMIC_BRAIN_FILE, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                logger.info(f"Loaded cosmic brain data from {COSMIC_BRAIN_FILE}.")
                return data
            else:
                logger.info(f"No existing cosmic brain data found. Starting fresh.")
                return {}
        except Exception as e:
            logger.error(f"Error loading cosmic brain data: {e}")
            return {}
    
    def start_audio_capture(self, duration=1.0, sample_rate=44100):
        """
        Starts the audio capture and processing threads.
        """
        if sd is None:
            logger.error("sounddevice is not installed. Cannot start audio capture.")
            return
        if not self.audio_running:
            self.audio_running = True
            self.audio_thread = threading.Thread(target=audio_capture_thread,
                                                 args=(self, self.audio_queue, duration, sample_rate),
                                                 daemon=True)
            self.audio_thread.start()
            self.processing_thread = threading.Thread(target=process_audio,
                                                      args=(self, self.audio_queue, self.processed_audio_queue, sample_rate),
                                                      daemon=True)
            self.processing_thread.start()
            logger.info("Started audio capture and processing threads.")
    
    def stop_audio_capture(self):
        """
        Stops the audio capture and processing threads.
        """
        if self.audio_running:
            self.audio_running = False
            self.audio_queue.put(None)
            self.processed_audio_queue.put(None)
            logger.info("Stopped audio capture.")
    
    def get_latest_frequency_data(self):
        """
        Retrieves the latest processed frequency data from the queue.
        """
        try:
            if not self.processed_audio_queue.empty():
                frequency_data = self.processed_audio_queue.get()
                return frequency_data
            else:
                return None
        except Exception as e:
            logger.error(f"Error retrieving frequency data: {e}")
            return None
    
    # Camera integration methods
    def start_camera_capture(self, device=0):
        """
        Starts the camera capture and processing threads.
        """
        if cv2 is None:
            logger.error("OpenCV is not installed. Camera capture is unavailable.")
            return
        if not self.camera_running:
            self.camera_running = True
            self.camera_queue = queue.Queue()
            self.processed_camera_queue = queue.Queue()
            self.camera_thread = threading.Thread(target=camera_capture_thread, args=(self, self.camera_queue, device), daemon=True)
            self.camera_thread.start()
            threading.Thread(target=process_camera, args=(self, self.camera_queue, self.processed_camera_queue), daemon=True).start()
            logger.info("Started camera capture and processing threads.")
    
    def stop_camera_capture(self):
        """
        Stops the camera capture and processing threads.
        """
        if self.camera_running:
            self.camera_running = False
    
    def get_latest_camera_data(self):
        """
        Retrieves the latest processed camera data from the queue.
        """
        try:
            if not self.processed_camera_queue.empty():
                return self.processed_camera_queue.get()
            else:
                return None
        except Exception as e:
            logger.error(f"Error retrieving camera data: {e}")
            return None

##############################
# CAMERA CAPTURE FUNCTIONS
##############################
def camera_capture_thread(simulator, camera_queue, device=0):
    """
    Captures frames from the camera and places them in the queue.
    """
    if cv2 is None:
        logger.error("OpenCV is not available.")
        return
    cap = cv2.VideoCapture(device)
    if not cap.isOpened():
        logger.error("Failed to open camera.")
        return
    while simulator.camera_running:
        ret, frame = cap.read()
        if not ret:
            logger.error("Failed to capture camera frame.")
            continue
        # Convert frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Convert image to grayscale to simplify brightness extraction
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Compute average brightness as a proxy for light data
        avg_brightness = np.mean(gray_frame)
        # Map brightness to a parameter (e.g., influencing particle behavior)
        brightness_param = avg_brightness / 255.0
        frequency_data = {'brightness': brightness_param}
        camera_queue.put(frequency_data)
        # Save frequency data to cosmic brain
        save_cosmic_brain(frequency_data)
        time.sleep(0.1)  # Adjust as needed for real-time responsiveness
    cap.release()

def process_camera(simulator, camera_queue, processed_camera_queue):
    """
    Processes camera data and places the processed data in the queue.
    """
    while simulator.camera_running or not camera_queue.empty():
        try:
            data = camera_queue.get()
            processed_camera_queue.put(data)
        except Exception as e:
            logger.error(f"Error processing camera data: {e}")

##############################
# EXTRA MODE: Tkinter/Pygame/Speech/TTS
##############################
# Initialize TTS engine
if pyttsx3 is not None:
    try:
        tts_engine = pyttsx3.init()
    except Exception as e:
        logger.warning(f"Failed to initialize TTS engine: {e}")
        tts_engine = None
else:
    tts_engine = None

def create_log_window(log_queue) -> tk.Tk:
    """
    Creates a Tkinter window to display system logs.
    """
    root = tk.Tk()
    root.title("System Logs")
    text_area = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=60, height=30)
    text_area.pack(expand=True, fill=tk.BOTH)
    
    def update_log():
        while not log_queue.empty():
            log_message = log_queue.get()
            text_area.insert(tk.END, log_message + "\n")
            text_area.see(tk.END)
        root.after(100, update_log)
    
    root.after(100, update_log)
    return root

def rotate_3d_point(point, yaw, pitch, roll):
    """
    Rotates a 3D point around the origin using Euler angles (yaw, pitch, roll).
    Yaw: rotation around Y axis (left/right)
    Pitch: rotation around X axis (up/down)
    Roll: rotation around Z axis (tilt)
    """
    x, y, z = point
    
    # Yaw rotation (around Y axis)
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    x1 = x * cos_yaw - z * sin_yaw
    z1 = x * sin_yaw + z * cos_yaw
    y1 = y
    
    # Pitch rotation (around X axis)
    cos_pitch = math.cos(pitch)
    sin_pitch = math.sin(pitch)
    y2 = y1 * cos_pitch - z1 * sin_pitch
    z2 = y1 * sin_pitch + z1 * cos_pitch
    x2 = x1
    
    # Roll rotation (around Z axis)
    cos_roll = math.cos(roll)
    sin_roll = math.sin(roll)
    x3 = x2 * cos_roll - y2 * sin_roll
    y3 = x2 * sin_roll + y2 * cos_roll
    z3 = z2
    
    return np.array([x3, y3, z3])

def visualization_loop(simulator):
    """
    Runs the Pygame visualization loop, rendering particles and evolving the visual representation.
    """
    global audio_queue_extra, particles_extra
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Math-Driven Creative System - Camera Controls: Mouse Drag=Rotate, Wheel=Zoom, Arrow Keys=Rotate")
    clock = pygame.time.Clock()
    running = True
    fullscreen = False
    
    # Camera control variables
    camera_yaw = 0.0  # Horizontal rotation (left/right)
    camera_pitch = 0.0  # Vertical rotation (up/down)
    camera_roll = 0.0  # Tilt rotation
    camera_zoom = 1e-11  # Zoom level (scale factor)
    zoom_speed = 1.2  # Zoom multiplier per scroll
    rotation_speed = 0.01  # Rotation speed for keyboard
    mouse_rotation_speed = 0.005  # Rotation speed for mouse drag
    
    # Mouse state for dragging
    mouse_dragging = False
    last_mouse_pos = None
    mouse_button_down = False
    # Optionally load a sample image
    ba2_files = glob.glob(os.path.join(DATA_DIRECTORY, '**', '*.ba2'), recursive=True)
    image_data = None
    sample_image_path = os.path.join(DATA_DIRECTORY, "sample_image.jpg")
    if ba2_files and os.path.exists(sample_image_path):
        try:
            if Image is not None:
                image = Image.open(sample_image_path).resize((WINDOW_WIDTH, WINDOW_HEIGHT))
                image_data = np.array(image)
                logger.info("Loaded sample image for visualization.")
            else:
                image_data = None
        except Exception as e:
            logger.error(f"Error loading sample image: {e}")
            image_data = None
    while running:
        screen.fill((0, 0, 0))
        
        # Handle keyboard input for continuous rotation
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            camera_yaw -= rotation_speed
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            camera_yaw += rotation_speed
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            camera_pitch -= rotation_speed
        if keys[pygame.K_DOWN] or keys[pygame.K_s]:
            camera_pitch += rotation_speed
        if keys[pygame.K_q]:
            camera_roll -= rotation_speed
        if keys[pygame.K_e]:
            camera_roll += rotation_speed
        if keys[pygame.K_PLUS] or keys[pygame.K_EQUALS]:
            camera_zoom *= zoom_speed
        if keys[pygame.K_MINUS]:
            camera_zoom /= zoom_speed
        
        # Limit pitch to prevent gimbal lock
        camera_pitch = max(-math.pi / 2 + 0.1, min(math.pi / 2 - 0.1, camera_pitch))
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_F11:
                    fullscreen = not fullscreen
                    if fullscreen:
                        screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
                    else:
                        screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left mouse button
                    mouse_dragging = True
                    mouse_button_down = True
                    last_mouse_pos = pygame.mouse.get_pos()
                elif event.button == 4:  # Mouse wheel up
                    camera_zoom *= zoom_speed
                elif event.button == 5:  # Mouse wheel down
                    camera_zoom /= zoom_speed
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:  # Left mouse button
                    mouse_dragging = False
                    mouse_button_down = False
            elif event.type == pygame.MOUSEMOTION:
                if mouse_dragging and last_mouse_pos is not None:
                    current_mouse_pos = pygame.mouse.get_pos()
                    dx = current_mouse_pos[0] - last_mouse_pos[0]
                    dy = current_mouse_pos[1] - last_mouse_pos[1]
                    
                    # Rotate based on mouse movement
                    camera_yaw += dx * mouse_rotation_speed
                    camera_pitch += dy * mouse_rotation_speed
                    
                    # Limit pitch to prevent gimbal lock
                    camera_pitch = max(-math.pi / 2 + 0.1, min(math.pi / 2 - 0.1, camera_pitch))
                    
                    last_mouse_pos = current_mouse_pos
        
        # Get audio token from queue
        audio_token = None
        dominant_freq = 440.0  # Default
        if not audio_queue_extra.empty():
            token_or_freq = audio_queue_extra.get()
            if isinstance(token_or_freq, dict):
                audio_token = token_or_freq
                dominant_freq = audio_token.get("freqs", [440.0])[0] if len(audio_token.get("freqs", [])) > 0 else 440.0
        else:
                # Backward compatibility: old format was just frequency
                dominant_freq = float(token_or_freq)
        
        # Camera push detection (if camera available)
        camera_frame_curr = None
        push_events = []
        if hasattr(simulator, 'get_latest_camera_data'):
            camera_frame_curr = simulator.get_latest_camera_data()
            # Guard against None frames
            if camera_frame_curr is not None:
                if simulator._last_camera_frame is not None:
                    push_events = detect_camera_push(simulator._last_camera_frame, camera_frame_curr)
                    # Apply push to simulator particles
                    if push_events and hasattr(simulator, 'particles') and len(simulator.particles) > 0:
                        apply_camera_push_to_particles(simulator.particles, push_events, WINDOW_WIDTH, WINDOW_HEIGHT)
                simulator._last_camera_frame = camera_frame_curr
        
        # Update particles with 12D CST features and audio tokens
        update_particles_extra(simulator, dominant_freq, image_data, audio_token=audio_token)
        
        # Render extra mode particles (with camera rotation)
        for p in particles_extra:
            # Apply camera rotation to extra particles
            particle_pos_3d = np.array([p.x, p.y, p.z])
            rotated_pos = rotate_3d_point(particle_pos_3d, camera_yaw, camera_pitch, camera_roll)
            # Project to 2D with zoom
            x_2d = rotated_pos[0] * camera_zoom
            y_2d = rotated_pos[1] * camera_zoom
            screen_x = int(WINDOW_WIDTH / 2 + x_2d)
            screen_y = int(WINDOW_HEIGHT / 2 - y_2d)
            if 0 <= screen_x < WINDOW_WIDTH and 0 <= screen_y < WINDOW_HEIGHT:
                pygame.draw.circle(screen, p.color, (screen_x, screen_y), max(1, int(p.radius)))
        
        # Render simulator particles (main simulation particles) with camera rotation
        if len(simulator.particles) > 0 and len(simulator.history) > 0:
            step = min(len(simulator.history) - 1, len(simulator.history) - 1)
            positions = simulator.history[step]
            camera_pos = [0, 0, 0]  # Camera at origin, rotation handles viewing angle
            
            for i, particle in enumerate(simulator.particles):
                if i < len(positions):
                    try:
                        pos = positions[i]
                        # Validate position values
                        if not np.isfinite(pos).all():
                            continue
                        
                        # Apply camera rotation to 3D position
                        rotated_pos = rotate_3d_point(pos, camera_yaw, camera_pitch, camera_roll)
                        
                        # Project 3D to 2D with perspective (using rotated Z for depth)
                        # Simple perspective projection
                        depth_factor = 1.0 / (1.0 + abs(rotated_pos[2]) * 1e-12)  # Depth scaling
                        x = rotated_pos[0] * camera_zoom * depth_factor
                        y = rotated_pos[1] * camera_zoom * depth_factor
                        
                        # Validate x, y are finite
                        if not (np.isfinite(x) and np.isfinite(y)):
                            continue
                        
                        # Screen coordinates
                        screen_x = int(WINDOW_WIDTH / 2 + x)
                        screen_y = int(WINDOW_HEIGHT / 2 - y)
                        
                        # Validate screen coordinates are within reasonable bounds
                        if not (-WINDOW_WIDTH <= screen_x < WINDOW_WIDTH * 2 and -WINDOW_HEIGHT <= screen_y < WINDOW_HEIGHT * 2):
                            continue
                        
                        # Color based on frequency or energy
                        if particle.frequency > 0:
                            hue = (particle.frequency / 20000.0) % 1.0
                            if plt:
                                try:
                                    color_rgb = plt.cm.hsv(hue)[:3]
                                    color = tuple(int(c * 255) for c in color_rgb)
                                except:
                                    color = (255, 255, 255)
                            else:
                                color = (255, 255, 255)
                        else:
                            # Energy-based coloring
                            try:
                                max_energy = max([p.Ec for p in simulator.particles] + [1.0])
                                norm_energy = particle.Ec / max_energy if max_energy > 0 else 0.0
                                if plt and np.isfinite(norm_energy):
                                    color_rgb = plt.cm.viridis(norm_energy)[:3]
                                    color = tuple(int(c * 255) for c in color_rgb)
                                else:
                                    color = (255, 255, 255)
                            except:
                                color = (255, 255, 255)
                        
                        # Size based on mass
                        try:
                            max_mass = max([p.mass for p in simulator.particles] + [1e-20])
                            size = max(2, int(2 + (particle.mass / max_mass) * 8))
                            size = min(size, 20)  # Limit max size
                        except:
                            size = 3
                        
                        # Draw particle (only if coordinates are valid)
                        try:
                            pygame.draw.circle(screen, color, (screen_x, screen_y), size)
                        except (TypeError, ValueError) as e:
                            # Skip invalid particles
                            continue
                    except Exception as e:
                        # Skip particles with errors
                        continue
        
        # Display 12D CST metrics overlay
        try:
            font = pygame.font.Font(None, 20)
            font_small = pygame.font.Font(None, 16)
            
            # Get 12D CST metrics from simulator
            metrics_text = []
            if simulator and hasattr(simulator, 'particles') and len(simulator.particles) > 0:
                # Compute metrics
                total_energy = compute_total_energy(simulator.particles)
                virial_ratio = check_virial(simulator.particles)
                sync_order = compute_synchronization_order_parameter(simulator.particles)
                avg_x12 = np.mean([p.x12 for p in simulator.particles]) if hasattr(simulator.particles[0], 'x12') else 0.0
                avg_omega = np.mean([p.omega for p in simulator.particles]) if hasattr(simulator.particles[0], 'omega') else 0.0
                
                metrics_text = [
                    f"12D CST Metrics:",
                    f"Energy: {total_energy:.2e}",
                    f"Virial: {virial_ratio:.3f}",
                    f"Sync (r): {sync_order:.3f}",
                    f"Avg x12: {avg_x12:.3f}",
                    f"Avg Ω: {avg_omega:.3f}"
                ]
                
                # Audio token info
                if audio_token:
                    metrics_text.extend([
                        f"Audio: RMS={audio_token.get('rms', 0):.3f}",
                        f"Chaos={audio_token.get('chaos', 0):.3f}",
                        f"Freqs={len(audio_token.get('freqs', []))}"
                    ])
                
                # Camera push info
                if push_events:
                    metrics_text.append(f"Push Events: {len(push_events)}")
            
            # Render metrics (top-right corner)
            y_offset = 5
            x_offset = WINDOW_WIDTH - 250
            for i, text in enumerate(metrics_text):
                color = (255, 255, 255) if i == 0 else (200, 200, 200)
                text_font = font if i == 0 else font_small
                text_surface = text_font.render(text, True, color)
                screen.blit(text_surface, (x_offset, y_offset))
                y_offset += 18
        except Exception as e:
            logger.debug(f"Error rendering metrics overlay: {e}")
        
        # Display camera controls help text (small, in corner)
        try:
            font = pygame.font.Font(None, 20)
            help_text = [
                "Camera: Mouse Drag=Rotate | Wheel=Zoom | Arrow/WASD=Rotate | Q/E=Roll | +/- = Zoom"
            ]
            y_offset = 5
            for text in help_text:
                text_surface = font.render(text, True, (200, 200, 200))
                screen.blit(text_surface, (5, y_offset))
                y_offset += 20
        except:
            pass  # If font rendering fails, continue without help text
        
        pygame.display.flip()
        clock.tick(60)
    pygame.quit()

def audio_capture_thread_extra(simulator, audio_queue, duration, sample_rate):
    """
    Captures audio data and places it in the audio queue (Extra Mode).
    """
    try:
        while simulator.audio_running:
            if sd is None:
                logger.error("sounddevice is not available. Cannot capture audio.")
                break
            audio_data = sd.rec(int(sample_rate * duration), samplerate=sample_rate, channels=1, blocking=True)
            audio_queue.put(audio_data.flatten())
    except Exception as e:
        logger.error(f"Error in audio_capture_thread_extra: {e}")

def tokenize_audio_frame(audio_data, samplerate=AUDIO_SAMPLERATE, top_n=32):
    """
    Tokenize audio frame into structured object with frequencies, magnitudes, φ-harmonics, and metrics.
    
    Args:
        audio_data: numpy array of audio samples
        samplerate: sample rate in Hz
        top_n: number of top frequencies to extract (default 32)
    
    Returns:
        dict with keys: freqs, mags, harmonics, rms, spectral_centroid, chaos, entropy, timestamp
    """
    try:
        # FFT analysis
        fft_result = np.fft.rfft(audio_data)
        freqs = np.fft.rfftfreq(len(audio_data), 1 / samplerate)
        magnitudes = np.abs(fft_result)
        
        # Get top N frequencies
        top_indices = np.argsort(magnitudes)[-top_n:][::-1]
        top_freqs = freqs[top_indices]
        top_mags = magnitudes[top_indices]
        
        # Normalize magnitudes
        if np.max(top_mags) > 0:
            top_mags = top_mags / np.max(top_mags)
        
        # Generate φ-harmonic series (golden ratio harmonics)
        phi = (1 + math.sqrt(5)) / 2
        dominant_freq = top_freqs[0] if len(top_freqs) > 0 else 440.0
        harmonics = []
        for n in range(8):  # Generate 8 harmonics
            harmonic_freq = dominant_freq * (phi ** n)
            harmonics.append({
                "frequency": harmonic_freq,
                "order": n,
                "amplitude": top_mags[0] * (phi ** (-n)) if len(top_mags) > 0 else 0.0
            })
        
        # Compute RMS energy
        rms = np.sqrt(np.mean(audio_data**2))
        
        # Compute spectral centroid
        if np.sum(magnitudes) > 0:
            spectral_centroid = np.sum(freqs * magnitudes) / np.sum(magnitudes)
        else:
            spectral_centroid = 0.0
        
        # Compute chaos metric (spectral spread)
        if np.sum(magnitudes) > 0:
            spectral_spread = np.sqrt(np.sum(((freqs - spectral_centroid)**2) * magnitudes) / np.sum(magnitudes))
            chaos = spectral_spread / (spectral_centroid + 1e-10)
        else:
            chaos = 0.0
        
        # Compute spectral entropy
        magnitude_norm = magnitudes / (np.sum(magnitudes) + 1e-12)
        spectral_entropy = -np.sum(magnitude_norm * np.log(magnitude_norm + 1e-12))
        
        token = {
            "freqs": top_freqs.tolist(),
            "mags": top_mags.tolist(),
            "harmonics": harmonics,
            "rms": float(rms),
            "spectral_centroid": float(spectral_centroid),
            "chaos": float(chaos),
            "entropy": float(spectral_entropy),
            "timestamp": time.time()
        }
        
        return token
    except Exception as e:
        logger.error(f"Error tokenizing audio frame: {e}")
        return {
            "freqs": [],
            "mags": [],
            "harmonics": [],
            "rms": 0.0,
            "spectral_centroid": 0.0,
            "chaos": 0.0,
            "entropy": 0.0,
            "timestamp": time.time()
        }


def audio_callback_extra(indata, frames, time_info, status):
    """
    Callback function for audio input stream in extra mode.
    Now tokenizes audio frames with full feature extraction.
    """
    global audio_queue_extra, log_queue_extra, frequency_history
    if status:
        log_queue_extra.put(f"Audio error: {status}")
        return
    try:
        # Handle both 1D and 2D arrays
        if hasattr(indata, 'shape'):
            if len(indata.shape) == 1:
                audio_data = indata
            else:
                audio_data = indata[:, 0]
        else:
            # Fallback if indata is not a numpy array
            audio_data = np.array(indata).flatten()
        
        # Tokenize audio frame
        token = tokenize_audio_frame(audio_data, AUDIO_SAMPLERATE, top_n=32)
        
        # Put token in queue
        audio_queue_extra.put(token)
        
        # Also extract dominant frequency for backward compatibility
        dominant_freq = token["freqs"][0] if len(token["freqs"]) > 0 else 440.0
        log_queue_extra.put(f"Audio Token: {len(token['freqs'])} freqs, RMS={token['rms']:.3f}, Chaos={token['chaos']:.3f}")
        frequency_history.append(dominant_freq)
    except Exception as e:
        log_queue_extra.put(f"Error in audio_callback_extra: {e}")

def detect_camera_push(camera_frame_prev, camera_frame_curr, threshold=10.0):
    """
    Detect optical flow or brightness changes in camera frames to generate push events.
    
    Args:
        camera_frame_prev: previous camera frame (numpy array)
        camera_frame_curr: current camera frame (numpy array)
        threshold: motion detection threshold
    
    Returns:
        list of push event dicts with keys: region, vector, magnitude, affected_particle_ids
    """
    try:
        if camera_frame_prev is None or camera_frame_curr is None:
            return []
        
        push_events = []
        
        # Convert to grayscale if needed
        if len(camera_frame_prev.shape) == 3:
            gray_prev = cv2.cvtColor(camera_frame_prev, cv2.COLOR_BGR2GRAY) if cv2 is not None else np.mean(camera_frame_prev, axis=2)
            gray_curr = cv2.cvtColor(camera_frame_curr, cv2.COLOR_BGR2GRAY) if cv2 is not None else np.mean(camera_frame_curr, axis=2)
        else:
            gray_prev = camera_frame_prev
            gray_curr = camera_frame_curr
        
        # Compute frame difference
        frame_diff = np.abs(gray_curr.astype(float) - gray_prev.astype(float))
        
        # Detect motion regions
        motion_mask = frame_diff > threshold
        
        if np.any(motion_mask):
            # Find motion regions
            if cv2 is not None:
                contours, _ = cv2.findContours(motion_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    if cv2.contourArea(contour) > 100:  # Minimum area threshold
                        M = cv2.moments(contour)
                        if M["m00"] > 0:
                            cx = int(M["m10"] / M["m00"])
                            cy = int(M["m01"] / M["m00"])
                            
                            # Compute motion vector (simplified - use centroid shift)
                            region = {"x": cx, "y": cy, "area": cv2.contourArea(contour)}
                            
                            # Estimate push vector (normalized to [-1, 1])
                            magnitude = np.mean(frame_diff[motion_mask])
                            vector = np.array([0.0, 0.0, 0.0])  # Placeholder - could use optical flow
                            
                            push_event = {
                                "region": region,
                                "vector": vector.tolist(),
                                "magnitude": float(magnitude),
                                "affected_particle_ids": []  # Will be filled based on proximity
                            }
                            push_events.append(push_event)
            else:
                # Fallback: simple brightness change detection
                brightness_change = np.mean(frame_diff)
                if brightness_change > threshold:
                    push_event = {
                        "region": {"x": gray_curr.shape[1]//2, "y": gray_curr.shape[0]//2, "area": gray_curr.size},
                        "vector": [0.0, 0.0, 0.0],
                        "magnitude": float(brightness_change),
                        "affected_particle_ids": []
                    }
                    push_events.append(push_event)
        
        return push_events
    except Exception as e:
        logger.error(f"Error detecting camera push: {e}")
        return []


def apply_camera_push_to_particles(particles, push_events, screen_width=WINDOW_WIDTH, screen_height=WINDOW_HEIGHT):
    """
    Apply camera push events to particles, creating force impulses.
    
    Args:
        particles: list of Particle objects
        push_events: list of push event dicts from detect_camera_push
        screen_width: screen width for coordinate mapping
        screen_height: screen height for coordinate mapping
    """
    try:
        for push_event in push_events:
            region = push_event["region"]
            vector = np.array(push_event["vector"])
            magnitude = push_event["magnitude"]
            
            # Map screen coordinates to 3D space (simplified mapping)
            push_x = (region["x"] / screen_width - 0.5) * 20.0
            push_y = (0.5 - region["y"] / screen_height) * 20.0
            push_pos = np.array([push_x, push_y, 0.0])
            
            # Apply push to nearby particles
            for particle in particles:
                if hasattr(particle, 'position'):
                    particle_pos = particle.position
                elif hasattr(particle, 'x'):
                    particle_pos = np.array([particle.x, particle.y, particle.z])
                else:
                    continue
                
                distance = np.linalg.norm(particle_pos - push_pos)
                push_radius = 5.0  # Push radius
                
                if distance < push_radius:
                    # Compute push force (inverse square law)
                    force_magnitude = magnitude * (1.0 / (distance + 1e-10)) * 0.1
                    push_direction = (particle_pos - push_pos) / (distance + 1e-10)
                    
                    # Apply impulse to particle velocity
                    if hasattr(particle, 'velocity'):
                        particle.velocity += push_direction * force_magnitude
                    elif hasattr(particle, 'dx'):
                        particle.dx += push_direction[0] * force_magnitude
                        particle.dy += push_direction[1] * force_magnitude
                        particle.dz += push_direction[2] * force_magnitude
                    
                    push_event["affected_particle_ids"].append(getattr(particle, 'id', id(particle)))
    except Exception as e:
        logger.error(f"Error applying camera push to particles: {e}")


def apply_tokens_to_particle(particle, token):
    """
    Apply audio token features to a particle, affecting its 12D CST properties.
    
    Args:
        particle: Particle object
        token: audio token dict from tokenize_audio_frame
    """
    try:
        if not hasattr(particle, 'x12'):
            return  # Not a 12D CST particle
        
        # Map RMS energy to particle energy
        if token.get("rms", 0) > 0:
            energy_boost = token["rms"] * 0.01
            particle.Ec += energy_boost
            particle.update_vi()
        
        # Map spectral centroid to x12 influence
        if len(token.get("freqs", [])) > 0:
            centroid_norm = token.get("spectral_centroid", 0) / 10000.0  # Normalize
            particle.x12 += centroid_norm * 0.1
            particle.x12 = max(-1.0, min(1.0, particle.x12))
        
        # Map chaos to phase perturbation
        chaos_factor = token.get("chaos", 0)
        particle.theta += chaos_factor * 0.1
        particle.theta = particle.theta % (2 * math.pi)
        
        # Map harmonics to omega boost
        if len(token.get("harmonics", [])) > 0:
            harmonic_boost = sum(h["amplitude"] for h in token["harmonics"]) * 0.01
            particle.omega += harmonic_boost
            particle.omega = max(0.0, particle.omega)
    except Exception as e:
        logger.error(f"Error applying tokens to particle: {e}")


def update_particles_extra(simulator, dominant_freq, image_data=None, feedback=None, audio_token=None):
    """
    Updates particles in the extra visualization mode based on dominant frequency and image data.
    Now integrates 12D CST features and audio tokenization.
    """
    global particles_extra
    if feedback is None:
        feedback = {
            "particle_updates": {
                "spawn_multiplier": 1.0,
                "color_adjustments": [0, 0, 0],
                "lifespan_multiplier": 1.0,
                "motion_modifier": 1.0
            }
        }
    
    # 12D CST Integration: Update simulator particles with full CST dynamics
    if simulator and hasattr(simulator, 'particles') and len(simulator.particles) > 0:
        dt = getattr(simulator, 'dt', 0.1)
        
        # Compute connectivity with similarity weighting
        Omega = simulator.network.compute_connectivity(sigma=0.5, m0=1.0)
        
        # Update 12D CST states for each simulator particle
        Ksync = 0.1  # Synchronization coupling
        for particle in simulator.particles:
            # Update internal adaptive states
            particle.update_x12(dt, k=0.1, gamma=0.01)
            particle.update_memory_state(dt, alpha=0.05)
            
            # Update phase synchronization
            neighbors = simulator.find_neighbors(particle) if hasattr(simulator, 'find_neighbors') else []
            particle.update_phase(dt, neighbors, Ksync=Ksync)
            
            # Apply audio token effects if provided
            if audio_token:
                apply_tokens_to_particle(particle, audio_token)
        
        # Compute Psi with full breakdown for extra mode
        if len(simulator.particles) > 0:
            Ec = np.array([p.Ec for p in simulator.particles])
            positions = np.array([p.position for p in simulator.particles])
            masses = np.array([p.mass for p in simulator.particles])
            psi_result = simulator.dynamics.compute_Psi(Ec, Omega, positions, masses, 
                                                       particles=simulator.particles, dt=dt)
            if isinstance(psi_result, dict):
                simulator.psi_breakdown = psi_result
        
        # Update entropy
        compute_shannon_entropy(simulator.particles, bins=32)
        
        # Adaptive timestep for extra mode
        adaptive_dt = compute_adaptive_dt(simulator.particles, dt_max=dt, safety_factor=0.1)
        if adaptive_dt < dt:
            simulator.dt = adaptive_dt
    
    # Visual particle spawning and updates (existing logic)
    spawn_multiplier = feedback["particle_updates"].get("spawn_multiplier", 1.0)
    particle_limit = MAX_EXTRA_PARTICLES - len(particles_extra)
    
    # Spawn visual particles based on audio token or dominant frequency
    spawn_count = int(50 * spawn_multiplier)
    if audio_token:
        # Spawn based on RMS energy, but clamp to reasonable values
        rms_factor = audio_token.get("rms", 0)
        spawn_count = int(spawn_count * (1.0 + min(rms_factor * 10, 5.0)))  # Cap at 5x multiplier
        # Accelerate decay when RMS is low
        if rms_factor < 0.01:
            for p in particles_extra:
                p.life -= 2  # Faster decay when quiet
    
    # Clamp spawn count to prevent excessive spawning
    spawn_count = min(spawn_count, 100, particle_limit)
    
    for _ in range(spawn_count):
        x, y, z = np.random.uniform(-10, 10, 3)
        color = [random.randint(0, 255) for _ in range(3)]
        particles_extra.append(VisualParticleExtra(x, y, z, color, random.randint(50, 100), random.uniform(0.5, 1.5)))
    
    if image_data is not None:
        for i, p in enumerate(particles_extra):
            if i < image_data.shape[0] * image_data.shape[1]:
                x, y = divmod(i, image_data.shape[1])
                color = image_data[x % image_data.shape[0], y % image_data.shape[1]]
                p.color = color.tolist()
                p.update(0, 0, 0, dominant_freq, target_position=(x, y, 0))
            else:
                dx, dy, dz = np.random.uniform(-0.1, 0.1, 3)
                p.update(dx, dy, dz, dominant_freq)
    else:
        for p in particles_extra:
            dx, dy, dz = np.random.uniform(-0.1, 0.1, 3)
            p.update(dx, dy, dz, dominant_freq)
    particles_extra[:] = [p for p in particles_extra if p.life > 0]
    math_system.evolve(dominant_freq, feedback)

def recognize_speech_extra():
    """
    Recognizes speech from the microphone and processes it in extra mode.
    """
    global log_queue_extra
    if sr is None:
        log_queue_extra.put("Speech recognition is not available. Install speech_recognition package.")
        return ""
    try:
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            print("Listening...")
            audio = recognizer.listen(source)
            try:
                text = recognizer.recognize_google(audio)
                log_queue_extra.put(f"Recognized Speech: {text}")
                learn_from_text_extra(text)
                return text
            except sr.UnknownValueError:
                log_queue_extra.put("Speech Recognition could not understand audio")
            except sr.RequestError as e:
                log_queue_extra.put(f"Speech Recognition error: {e}")
    except Exception as e:
        log_queue_extra.put(f"Error in speech recognition: {e}")
    return ""

def speak_text_extra(text: str):
    """
    Uses TTS to speak the provided text in extra mode.
    """
    global tts_engine
    if tts_engine is None:
        logger.warning("TTS engine is not available. Cannot speak text.")
        return
    try:
        tts_engine.say(text)
        tts_engine.runAndWait()
    except Exception as e:
        logger.error(f"Error in TTS: {e}")

def learn_from_text_extra(text: str):
    """
    Processes learned text by updating the learning data in extra mode.
    """
    global learning_data, log_queue_extra
    try:
        learning_data[text] = f"Learned response to: {text}"
        save_learning_data_extra()
        speak_text_extra(f"I have learned {text}")
    except Exception as e:
        log_queue_extra.put(f"Learning error: {e}")

def save_learning_data_extra():
    """
    Saves the learning data to a JSON file in extra mode.
    """
    global learning_data
    try:
        with open("learning_data.json", "w", encoding="utf-8") as f:
            json.dump(learning_data, f, indent=4)
        logger.info("Learning data saved.")
    except Exception as e:
        logger.error(f"Error saving learning data: {e}")

def load_learning_data_extra():
    """
    Loads learning data from a JSON file in extra mode.
    """
    global learning_data
    try:
        with open("learning_data.json", "r", encoding="utf-8") as f:
            learning_data = json.load(f)
        logger.info("Learning data loaded.")
    except Exception:
        learning_data = {}
        logger.warning("No existing learning data found. Starting fresh.")

##############################
# VISUALIZER CLASS (Streamlit & Mayavi)
##############################
# Note: The class VisualParticleExtra is defined above and used in extra mode.

# Globals for extra mode
audio_queue_extra = queue.Queue()
log_queue_extra = queue.Queue()
particles_extra: List[VisualParticleExtra] = []
frequency_history = collections.deque(maxlen=30)
learning_data = {}
audio_stream_global = None  # Global audio stream for extra mode microphone reactivity
XAI_API_KEY = "your-api-key-here"  # Placeholder for any future AI integrations

##############################
# STREAMLIT INTERFACE
##############################
def run_streamlit_ui():
    """
    Runs the Streamlit user interface for the simulation.
    """
    st.set_page_config(page_title="Cosmic Synapse Theory Simulation", layout="wide")
    st.title("Cosmic Synapse Theory (Madsen's Theory) Simulation")
    st.sidebar.header("Simulation Controls")
    
    # Simulation Parameters
    num_particles = st.sidebar.slider("Number of Particles", 100, 1000, 100, step=100)
    steps = st.sidebar.slider("Number of Steps", 100, 5000, 1000, step=100)
    dt = st.sidebar.slider("Time Step (s)", 0.1, 10.0, 1.0, step=0.1)
    E_replicate_slider = st.sidebar.slider("Replication Energy Threshold (J)", 1e40, 1e60, E_replicate, step=1e40, format="%.0e")
    alpha = st.sidebar.slider("Alpha (J/m)", 1e-12, 1e-8, alpha_initial, step=1e-10, format="%.1e")
    lambda_evo = st.sidebar.slider("Lambda Evolution (J)", 0.1, 10.0, lambda_evo_initial, step=0.1)
    data_directory = st.sidebar.text_input("Data Directory Path", DATA_DIRECTORY)
    audio_duration = st.sidebar.slider("Audio Capture Duration (s)", 0.5, 5.0, 1.0, step=0.5)
    sample_rate = st.sidebar.slider("Audio Sample Rate (Hz)", 8000, 48000, 44100, step=1000)
    
    global MAX_PARTICLES
    MAX_PARTICLES = num_particles * 10  # Adjust maximum particles based on user input
    
    # Initialize Simulator and Neural Network
    global SimulatorInstance
    if "simulator" not in st.session_state:
        SimulatorInstance = Simulator(num_particles=num_particles, steps=steps, dt=dt, data_directory=data_directory)
        st.session_state.simulator = SimulatorInstance
        st.session_state.neural_net = ParticleNeuralNet()
        st.session_state.neural_net.eval()
        for param in st.session_state.neural_net.parameters():
            nn.init.uniform_(param, -0.1, 0.1)
        st.session_state.running = False
        st.session_state.step = 0
        st.session_state.audio_running = False
        logger.info("Simulation and neural network initialized.")
    else:
        SimulatorInstance = st.session_state.simulator
    
    neural_net = st.session_state.neural_net
    
    # Update Simulation Parameters based on user input
    SimulatorInstance.replication.E_replicate = E_replicate_slider
    SimulatorInstance.dynamics.alpha = alpha
    SimulatorInstance.dynamics.lambda_evo_initial = lambda_evo  # Update initial lambda
    SimulatorInstance.data_directory = data_directory
    
    # Control Buttons
    start_button = st.sidebar.button("Start Simulation")
    pause_button = st.sidebar.button("Pause Simulation")
    reset_button = st.sidebar.button("Reset Simulation")
    run_full_button = st.sidebar.button("Run Full Simulation")
    load_data_button = st.sidebar.button("Load Simulation Data")
    visualize_metrics_button = st.sidebar.button("Visualize Metrics")
    animate_simulation_button = st.sidebar.button("Animate Simulation")
    volumetric_heatmap_button = st.sidebar.button("Volumetric Heatmap")
    start_audio_button = st.sidebar.button("Start Audio Capture")
    stop_audio_button = st.sidebar.button("Stop Audio Capture")
    
    # ReplayManager controls
    st.sidebar.subheader("Replay Manager")
    if "replay_manager" not in st.session_state:
        st.session_state.replay_manager = ReplayManager()
    record_button = st.sidebar.button("Start Recording")
    stop_record_button = st.sidebar.button("Stop Recording")
    replay_button = st.sidebar.button("Start Replay")
    stop_replay_button = st.sidebar.button("Stop Replay")
    
    # Handle Control Buttons
    if start_button:
        st.session_state.running = True
        st.sidebar.write("Simulation Started.")
    if pause_button:
        st.session_state.running = False
        st.sidebar.write("Simulation Paused.")
    if reset_button:
        simulator = Simulator(num_particles=num_particles, steps=steps, dt=dt, data_directory=data_directory)
        st.session_state.simulator = simulator
        SimulatorInstance = simulator
        st.session_state.neural_net = ParticleNeuralNet()
        st.session_state.neural_net.eval()
        for param in st.session_state.neural_net.parameters():
            nn.init.uniform_(param, -0.1, 0.1)
        st.session_state.running = False
        st.session_state.step = 0
        st.sidebar.write("Simulation Reset.")
        logger.info("Simulation reset.")
    if start_audio_button:
        if not SimulatorInstance.audio_running:
            SimulatorInstance.start_audio_capture(duration=audio_duration, sample_rate=sample_rate)
            st.sidebar.write("Audio Capture Started.")
            st.session_state.audio_running = True
        else:
            st.sidebar.write("Audio Capture is already running.")
    if stop_audio_button:
        if SimulatorInstance.audio_running:
            SimulatorInstance.stop_audio_capture()
            st.sidebar.write("Audio Capture Stopped.")
            st.session_state.audio_running = False
        else:
            st.sidebar.write("Audio Capture is not running.")
    
    # Handle ReplayManager controls
    if record_button:
        st.session_state.replay_manager.start_recording()
        st.sidebar.write("Recording started.")
    if stop_record_button:
        st.session_state.replay_manager.stop_recording()
        st.sidebar.write(f"Recording stopped. {len(st.session_state.replay_manager.recorded_frames)} frames recorded.")
    if replay_button:
        st.session_state.replay_manager.start_replay()
        st.sidebar.write("Replay started.")
    if stop_replay_button:
        st.session_state.replay_manager.stop_replay()
        st.sidebar.write("Replay stopped.")
    
    # Create placeholder containers for real-time updates
    if "plot_container" not in st.session_state:
        st.session_state.plot_container = st.empty()
    if "metrics_container" not in st.session_state:
        st.session_state.metrics_container = st.empty()
    
    # Real-time simulation loop
    if st.session_state.running and st.session_state.step < SimulatorInstance.steps:
        # Get frequency data - use replay if active, otherwise live audio
        if st.session_state.replay_manager.is_replaying:
            replay_frame = st.session_state.replay_manager.get_next_frame()
            if replay_frame is not None:
                frequency_data = replay_frame
            else:
                # Replay ended, stop replay
                st.session_state.replay_manager.stop_replay()
                frequency_data = SimulatorInstance.get_latest_frequency_data()
        else:
            frequency_data = SimulatorInstance.get_latest_frequency_data()
            # Record frame if recording
            if st.session_state.replay_manager.is_recording:
                st.session_state.replay_manager.record_frame(frequency_data)
        
        camera_data = SimulatorInstance.get_latest_camera_data()
        SimulatorInstance.run_step(neural_net, frequency_data, camera_data)
        # Sync step counter with simulator
        st.session_state.step = SimulatorInstance.step
        
        # Update 3D plot in real-time
        with st.session_state.plot_container.container():
            st.subheader("🌌 Real-Time 3D Particle Visualization")
            SimulatorInstance.visualizer.plot_particles(
                len(SimulatorInstance.history) - 1, 
                frequency_data=frequency_data,
                plot_container=None
            )
        
        # Update metrics in real-time
        with st.session_state.metrics_container.container():
            st.subheader("📊 Real-Time Metrics")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Simulation Step", f"{st.session_state.step}/{SimulatorInstance.steps}")
            with col2:
                st.metric("Particles", len(SimulatorInstance.particles))
            with col3:
                if SimulatorInstance.total_energy_history:
                    st.metric("Total Energy", f"{SimulatorInstance.total_energy_history[-1]:.2e} J")
            with col4:
                if SimulatorInstance.entropy_history:
                    st.metric("Avg Entropy", f"{SimulatorInstance.entropy_history[-1]:.2e}")
            
            # Additional metrics
            if SimulatorInstance.kinetic_energy_history:
                st.write(f"**Kinetic Energy:** {SimulatorInstance.kinetic_energy_history[-1]:.2e} J")
            if SimulatorInstance.potential_energy_history:
                st.write(f"**Potential Energy:** {SimulatorInstance.potential_energy_history[-1]:.2e} J")
        
            # 12D CST Metrics
            st.subheader("🔬 12D Cosmic Synapse Theory Metrics")
            if len(SimulatorInstance.particles) > 0 and hasattr(SimulatorInstance.particles[0], 'x12'):
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    avg_x12 = np.mean([p.x12 for p in SimulatorInstance.particles])
                    st.metric("Avg x12", f"{avg_x12:.4f}")
                with col2:
                    avg_omega = np.mean([p.omega for p in SimulatorInstance.particles])
                    st.metric("Avg Ω", f"{avg_omega:.4f}")
                with col3:
                    sync_order = compute_synchronization_order_parameter(SimulatorInstance.particles)
                    st.metric("Sync Order (r)", f"{sync_order:.4f}")
                with col4:
                    virial_ratio = check_virial(SimulatorInstance.particles)
                    st.metric("Virial Ratio", f"{virial_ratio:.4f}")
                
                # Conservation diagnostics
                total_energy = compute_total_energy(SimulatorInstance.particles)
                total_momentum = compute_total_momentum(SimulatorInstance.particles)
                total_angular_momentum = compute_total_angular_momentum(SimulatorInstance.particles)
                
                # Track energy history for drift calculation
                if "energy_history" not in st.session_state:
                    st.session_state.energy_history = []
                st.session_state.energy_history.append(total_energy)
                if len(st.session_state.energy_history) > 100:
                    st.session_state.energy_history = st.session_state.energy_history[-100:]
                
                # Calculate energy drift
                energy_drift = 0.0
                if len(st.session_state.energy_history) > 1:
                    initial_energy = st.session_state.energy_history[0]
                    if abs(initial_energy) > 1e-10:
                        energy_drift = abs((total_energy - initial_energy) / initial_energy) * 100.0
                
                st.write("**Conservation Diagnostics:**")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Energy", f"{total_energy:.2e} J", 
                             delta=f"{energy_drift:.2f}% drift" if energy_drift > 0.01 else None)
                with col2:
                    momentum_mag = np.linalg.norm(total_momentum)
                    st.metric("Total Momentum", f"{momentum_mag:.2e} kg·m/s")
                with col3:
                    ang_momentum_mag = np.linalg.norm(total_angular_momentum)
                    st.metric("Total Angular Momentum", f"{ang_momentum_mag:.2e} kg·m²/s")
                with col4:
                    st.metric("Virial Ratio", f"{virial_ratio:.4f}", 
                             delta="Virialized" if 0.95 < virial_ratio < 1.05 else "Not virialized")
                
                # ψ Breakdown
                if hasattr(SimulatorInstance, 'psi_breakdown') and SimulatorInstance.psi_breakdown:
                    st.subheader("⚛️ Psi (ψ) Breakdown")
                    psi_breakdown = SimulatorInstance.psi_breakdown
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Normalized Terms:**")
                        st.write(f"- φ·Ec/c²: {np.mean(psi_breakdown.get('phi_term', [0])):.4e}")
                        st.write(f"- λ term: {np.mean(psi_breakdown.get('lambda_term', [0])):.4e}")
                        st.write(f"- Velocity term: {np.mean(psi_breakdown.get('velocity_term', [0])):.4e}")
                        st.write(f"- x12 term: {np.mean(psi_breakdown.get('x12_term', [0])):.4e}")
                    with col2:
                        st.write("**Energy Terms:**")
                        st.write(f"- Synaptic (Ω·Ec): {np.mean(psi_breakdown.get('synaptic_term', [0])):.4e}")
                        st.write(f"- Gravitational: {np.mean(psi_breakdown.get('grav_term', [0])):.4e}")
                        st.write(f"- Dark Matter: {np.mean(psi_breakdown.get('dm_term', [0])):.4e}")
                        st.write(f"- **Total ψ:** {np.mean(psi_breakdown.get('psi_total', [0])):.4e}")
            
            # Audio Token Stream
            if "audio_tokens" not in st.session_state:
                st.session_state.audio_tokens = []
            
            # Get latest audio token if available
            if hasattr(SimulatorInstance, 'get_latest_frequency_data'):
                freq_data = SimulatorInstance.get_latest_frequency_data()
                # Tokenize if we have raw audio data
                if freq_data is not None and isinstance(freq_data, np.ndarray) and len(freq_data) > 0:
                    try:
                        token = tokenize_audio_frame(freq_data, AUDIO_SAMPLERATE, top_n=32)
                        st.session_state.audio_tokens.append(token)
                        # Keep only last 50 tokens
                        if len(st.session_state.audio_tokens) > 50:
                            st.session_state.audio_tokens = st.session_state.audio_tokens[-50:]
                    except:
                        pass
            
            # Entropy Chart (time series)
            if "entropy_history" not in st.session_state:
                st.session_state.entropy_history = []
            if len(SimulatorInstance.particles) > 0:
                current_entropy = compute_shannon_entropy(SimulatorInstance.particles, bins=32)
                st.session_state.entropy_history.append(current_entropy)
                # Keep only last 100 points
                if len(st.session_state.entropy_history) > 100:
                    st.session_state.entropy_history = st.session_state.entropy_history[-100:]
            
            if len(st.session_state.entropy_history) > 1:
                st.subheader("📈 Entropy Time Series")
                if pd is not None:
                    entropy_df = pd.DataFrame({
                        "Step": range(len(st.session_state.entropy_history)),
                        "Shannon Entropy": st.session_state.entropy_history
                    })
                    st.line_chart(entropy_df.set_index("Step"))
                else:
                    # Fallback: simple line chart using dict
                    st.line_chart({"Shannon Entropy": st.session_state.entropy_history})
            
            # Display token stream
            if st.session_state.audio_tokens:
                st.subheader("🎵 Audio Token Stream")
                with st.expander("View Token Stream", expanded=False):
                    # Show summary statistics
                    if len(st.session_state.audio_tokens) > 0:
                        avg_rms = np.mean([t.get("rms", 0) for t in st.session_state.audio_tokens])
                        avg_chaos = np.mean([t.get("chaos", 0) for t in st.session_state.audio_tokens])
                        avg_entropy = np.mean([t.get("entropy", 0) for t in st.session_state.audio_tokens])
                        st.metric("Avg RMS", f"{avg_rms:.4f}")
                        st.metric("Avg Chaos", f"{avg_chaos:.4f}")
                        st.metric("Avg Spectral Entropy", f"{avg_entropy:.4f}")
                    
                    # Show last 10 tokens
                    for i, token in enumerate(st.session_state.audio_tokens[-10:]):  # Show last 10
                        st.json({
                            "timestamp": token.get("timestamp", 0),
                            "freqs_count": len(token.get("freqs", [])),
                            "rms": token.get("rms", 0),
                            "spectral_centroid": token.get("spectral_centroid", 0),
                            "chaos": token.get("chaos", 0),
                            "entropy": token.get("entropy", 0),
                            "harmonics": len(token.get("harmonics", []))
                        })
            
            # Push Events Log (if available)
            if "push_events_log" not in st.session_state:
                st.session_state.push_events_log = []
            
            if len(st.session_state.push_events_log) > 0:
                st.subheader("📹 Camera Push Events")
                with st.expander("View Push Events", expanded=False):
                    for event in st.session_state.push_events_log[-10:]:  # Show last 10
                        st.json({
                            "region": event.get("region", {}),
                            "magnitude": event.get("magnitude", 0),
                            "affected_particles": len(event.get("affected_particle_ids", []))
                        })
        
        # Auto-refresh to continue simulation (throttled to avoid excessive reruns)
        if "last_rerun_time" not in st.session_state:
            st.session_state.last_rerun_time = time.time()
        
        current_time = time.time()
        if current_time - st.session_state.last_rerun_time >= 1.0:  # Throttle to 1 second
            st.session_state.last_rerun_time = current_time
            st.rerun()
        else:
            time.sleep(0.1)  # Small delay
    else:
        # Show current state when paused
        if len(SimulatorInstance.history) > 0:
            with st.session_state.plot_container.container():
                st.subheader("🌌 3D Particle Visualization (Paused)")
                SimulatorInstance.visualizer.plot_particles(
                    len(SimulatorInstance.history) - 1, 
                    frequency_data=None,
                    plot_container=None
                )
    
    # Run Full Simulation with real-time updates
    if run_full_button:
        st.session_state.running = True
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i in range(steps):
            if not st.session_state.running:
                break
                
            frequency_data = SimulatorInstance.get_latest_frequency_data()
            camera_data = SimulatorInstance.get_latest_camera_data()
            SimulatorInstance.run_step(neural_net, frequency_data, camera_data)
            st.session_state.step += 1
            
            # Update progress
            progress = (i + 1) / steps
            progress_bar.progress(progress)
            status_text.text(f"Running... Step {i+1}/{steps} | Particles: {len(SimulatorInstance.particles)}")
            
            # Update visualization every 10 steps for performance
            if (i + 1) % 10 == 0:
                with st.session_state.plot_container.container():
                    st.subheader("🌌 Real-Time 3D Particle Visualization")
                    SimulatorInstance.visualizer.plot_particles(
                        len(SimulatorInstance.history) - 1, 
                        frequency_data=frequency_data,
                        plot_container=None
                    )
                st.rerun()
        
        st.session_state.running = False
        st.success("Simulation completed.")
        SimulatorInstance.visualizer.plot_metrics()
    
    # Load Simulation Data
    if load_data_button:
        filename = st.sidebar.text_input("Enter filename to load", "simulation_data.pkl")
        if st.sidebar.button("Load"):
            if os.path.exists(filename):
                data = load_simulation_data(filename)
                SimulatorInstance.history = data.get('history', [])
                SimulatorInstance.total_energy_history = data.get('total_energy_history', [])
                SimulatorInstance.kinetic_energy_history = data.get('kinetic_energy_history', [])
                SimulatorInstance.potential_energy_history = data.get('potential_energy_history', [])
                SimulatorInstance.entropy_history = data.get('entropy_history', [])
                st.success("Simulation data loaded.")
                SimulatorInstance.visualizer.plot_metrics()
                SimulatorInstance.visualizer.animate_simulation()
            else:
                st.error("File not found.")
    
    # Visualize Metrics
    if visualize_metrics_button:
        SimulatorInstance.visualizer.plot_metrics()
    
    # Animate Simulation
    if animate_simulation_button:
        SimulatorInstance.visualizer.animate_simulation()
    
    # Plot Volumetric Heatmap
    if volumetric_heatmap_button:
        step_to_plot = st.number_input("Enter step number for volumetric heatmap", min_value=1, max_value=SimulatorInstance.steps, value=st.session_state.step)
        if st.button("Plot Volumetric Heatmap"):
            SimulatorInstance.visualizer.plot_volumetric_heatmap(step_to_plot)

##############################
# EXTRA MODE: Tkinter/Pygame/Speech/TTS
##############################
def run_extra_mode():
    """
    Runs the standalone mode with Tkinter and Pygame for enhanced visualization and interaction.
    """
    load_learning_data_extra()
    tk_log = create_log_window(log_queue_extra)
    # Initialize SimulatorInstance for extra mode
    global SimulatorInstance
    SimulatorInstance = Simulator()
    
    # Start audio capture for extra mode (microphone input)
    # Store audio_stream globally so it persists
    global audio_stream_global
    audio_stream_global = None
    if sd is not None:
        SimulatorInstance.start_audio_capture(duration=AUDIO_DURATION, sample_rate=AUDIO_SAMPLERATE)
        # Also start audio stream with callback for real-time processing
        try:
            audio_stream_global = sd.InputStream(samplerate=AUDIO_SAMPLERATE, channels=1, callback=audio_callback_extra)
            audio_stream_global.start()
            logger.info("Audio stream started for extra mode - microphone input will react to sound/blowing.")
        except Exception as e:
            logger.warning(f"Could not start audio stream: {e}")
    
    # Store audio_stream so it doesn't get garbage collected
    def cleanup_on_exit():
        global audio_stream_global
        if audio_stream_global is not None:
            try:
                audio_stream_global.stop()
                audio_stream_global.close()
            except:
                pass
    import atexit
    atexit.register(cleanup_on_exit)
    
    # Start visualization loop
    threading.Thread(target=visualization_loop, args=(SimulatorInstance,), daemon=True).start()
    
    # Start simulation update loop that processes audio and updates particles
    def simulation_update_loop():
        """Updates simulator with audio data in extra mode."""
        global SimulatorInstance
        neural_net = ParticleNeuralNet()
        neural_net.eval()
        while True:
            try:
                frequency_data = SimulatorInstance.get_latest_frequency_data()
                camera_data = SimulatorInstance.get_latest_camera_data()
                SimulatorInstance.run_step(neural_net, frequency_data, camera_data)
                time.sleep(0.1)  # Update every 100ms
            except Exception as e:
                logger.error(f"Error in simulation update loop: {e}")
                time.sleep(0.5)
    
    threading.Thread(target=simulation_update_loop, daemon=True).start()
    
    # Optionally start camera capture if available
    if cv2 is not None:
        SimulatorInstance.start_camera_capture(device=0)
    
    # Start speech recognition in a separate thread
    threading.Thread(target=speech_recognition_thread, args=(SimulatorInstance,), daemon=True).start()
    tk_log.mainloop()

def speech_recognition_thread(simulator):
    """
    Continuously listens for speech and processes it.
    """
    while True:
        text = recognize_speech_extra()
        if text:
            logger.info(f"Recognized and processed speech: {text}")

##############################
# MAIN EXECUTION
##############################
def main_entry():
    """
    Entry point of the program. Determines the mode to run based on command-line arguments.
    """
    if len(sys.argv) > 1 and sys.argv[1] == "tk":
        run_extra_mode()
    else:
        run_streamlit_ui()

if __name__ == "__main__":
    main_entry()
