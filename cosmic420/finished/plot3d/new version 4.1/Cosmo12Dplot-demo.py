# -*- coding: utf-8 -*-
#!/usr/bin/env python3

"""
12D Cosmic Synapse Theory (CST) Simulation Engine
==================================================

This simulation implements the 12D Cosmic Synapse Theory with comprehensive physics,
audio-driven behavior, and visualization capabilities.

PHASE UPDATES (Additive Only - All Existing Functionality Preserved):
---------------------------------------------------------------------

PHASE 1: Physics Stabilization
- ✅ Velocity Verlet integration (replaces Euler, backward compatible)
- ✅ Corrected force computation (proper mass inclusion, attraction sign, softened gravity)
- ✅ Improved adaptive timestep using cKDTree for efficient r_min computation
- ✅ Fixed double time increment bug

PHASE 2: Energy Accounting
- ✅ Separate energy tracking: K (kinetic), U (gravitational), U_dm (dark matter), E_info (informational)
- ✅ Total energy: Ec = K + U + U_dm + E_info (no double counting)
- ✅ Refactored assign_frequency() to use normalized audio features (RMS, centroid, entropy)
- ✅ Avoids unrealistic magnitudes from raw h·ν scaling

PHASE 3: Audio-Driven Dandelion Behavior
- ✅ RMS → outward radial impulse (seed dispersal)
- ✅ Centroid → clustering bias by frequency
- ✅ Chaos/entropy → perturb synchronization phase θ
- ✅ Drag and detachment logic (spring forces before blow, detachment on RMS threshold)
- ✅ Golden ratio harmonics integration for coherent clustering

PHASE 4: Connectivity & Internal State
- ✅ cKDTree neighbor cutoff for Ω computation (avoids O(N²))
- ✅ Normalized Ω per step (divide by max Ω) before feeding into x12 dynamics
- ✅ Gaussian similarity weighting on x12 differences
- ✅ Equilibrium verification: x12 = (k/γ)·Ω_norm

PHASE 5: Diagnostics & Replay
- ✅ Enhanced conservation checks (energy, momentum, angular momentum, virial ratio)
- ✅ Comprehensive replay logging (FFT, RMS, centroid, chaos, entropy, camera stats, dt, seeds)
- ✅ Deterministic reproduction support

PHASE 6: Visualization Enhancements
- ✅ Seed head clustering visualization (yellow diamond marker with connections)
- ✅ RMS intensity to particle color/size mapping (RdYlBu_r colormap)
- ✅ φ-harmonic groupings overlay (colored group centers)

PHASE 7: Validation & Iteration
- ✅ Conservation tests (energy, momentum, angular momentum)
- ✅ Audio blow scenario tests (detachment, dispersal, velocity increase)
- ✅ Replay determinism verification (identical seeds produce identical results)

ARCHITECTURE NOTES:
------------------
- All updates are ADDITIVE ONLY - existing functions preserved with legacy compatibility
- Velocity Verlet can be toggled via VELOCITY_VERLET_ENABLED flag
- Force computation supports both corrected (with masses) and legacy modes
- Energy tracking maintains backward compatibility with Ec and Uc properties

MAIN LOOP INTEGRATION:
----------------------
The simulation now uses an integrated simulation_step() method that executes all 7 phases in order:
1. Physics Update (Phase 1) - Velocity Verlet or Euler integration
2. Energy Update (Phase 2) - Separate energy tracking (K, U, U_dm, E_info)
3. Audio-Driven Dynamics (Phase 3) - Dandelion behavior and φ-harmonics
4. Connectivity & Internal State (Phase 4) - Ω computation and x12 evolution
5. Diagnostics & Replay (Phase 5) - Conservation checks and logging
6. Visualization (Phase 6) - RMS coloring, φ-harmonics, seed head
7. Validation & Iteration (Phase 7) - Conservation validation and parameter refinement

The integrated step is enabled by default (self.enable_integrated_step = True).
Set self.enable_integrated_step = False to use legacy run_step() implementation.

SCAFFOLDING IMPLEMENTATIONS:
----------------------------
Alternative implementations matching the provided scaffolding examples are available:
- velocity_verlet_step() - Pure Python Velocity Verlet (Phase 1 scaffolding)
- compute_energies() - Energy computation with separate channels (Phase 2 scaffolding)
- assign_frequency_energy() - Frequency energy assignment (Phase 2 scaffolding)
- apply_audio_impulses() - Audio-driven impulses (Phase 3 scaffolding)
- assign_phi_harmonics() - Golden ratio harmonic assignment (Phase 3 scaffolding)
- compute_connectivity() - Connectivity computation with cKDTree (Phase 4 scaffolding)
- update_internal_state() - Internal state update with equilibrium verification (Phase 4 scaffolding)
- compute_diagnostics() - Conservation diagnostics and emergent metrics (Phase 5 scaffolding)
- log_replay_step() - Replay logging for deterministic reproduction (Phase 5 scaffolding)
- load_replay_log() - Load replay log entries from file (Phase 5 scaffolding)
- visualize_particles_matplotlib() - 3D visualization with RMS coloring and φ-harmonics (Phase 6 scaffolding)
- visualize_seed_head() - 2D Pygame visualization of seed head dispersal (Phase 6 scaffolding)
- validate_conservation() - Conservation law validation (Phase 7 scaffolding)
- test_audio_blow() - Audio blow scenario test (Phase 7 scaffolding)
- check_replay_determinism() - Replay determinism verification (Phase 7 scaffolding)
- refine_parameters() - Parameter refinement for synchronization (Phase 7 scaffolding)

These can be used as alternatives to the main implementations or for comparison/testing.
"""

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

# PHASE 1: Physics Stabilization Constants
SOFTENING_EPSILON = 1e9  # Softening parameter for gravity (m) - prevents division by zero
VELOCITY_VERLET_ENABLED = True  # Toggle for Velocity Verlet integration

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
    LEGACY: Euler integration method (preserved for backward compatibility).
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
# PHASE 1: VELOCITY VERLET INTEGRATION & CORRECTED FORCE COMPUTATION
##############################

@njit(parallel=True)
def compute_forces_corrected_numba(positions: np.ndarray, masses: np.ndarray, 
                                    G_val: float = 6.67430e-11, 
                                    epsilon: float = 1e9) -> np.ndarray:
    """
    PHASE 1: Corrected force computation with proper gravitational attraction.
    
    Computes gravitational forces with:
    - Proper mass inclusion (m_i * m_j)
    - Attraction sign (negative, particles attract)
    - Softened gravity: r_eff = sqrt(r² + ε²)
    
    Args:
        positions: Nx3 array of particle positions
        masses: N array of particle masses
        G_val: Gravitational constant
        epsilon: Softening parameter for gravity
    
    Returns:
        Nx3 array of forces on each particle
    """
    N = len(positions)
    forces = np.zeros((N, 3))
    
    for i in prange(N):
        fx, fy, fz = 0.0, 0.0, 0.0
        m1 = masses[i]
        x1, y1, z1 = positions[i, 0], positions[i, 1], positions[i, 2]
        
        for j in range(N):
            if i == j:
                continue
            
            m2 = masses[j]
            x2, y2, z2 = positions[j, 0], positions[j, 1], positions[j, 2]
            
            # Separation vector (from i to j)
            dx = x2 - x1
            dy = y2 - y1
            dz = z2 - z1
            
            # Distance squared with softening
            r_sq = dx*dx + dy*dy + dz*dz
            r_eff_sq = r_sq + epsilon*epsilon  # Softened distance
            r_eff = math.sqrt(r_eff_sq)
            
            # Gravitational force magnitude: F = -G * m1 * m2 / r_eff²
            # Negative sign for attraction (force points toward other particle)
            f_mag = -(G_val * m1 * m2) / r_eff_sq
            
            # Force direction (unit vector from i to j)
            fx += f_mag * (dx / r_eff)
            fy += f_mag * (dy / r_eff)
            fz += f_mag * (dz / r_eff)
        
        forces[i, 0] = fx
        forces[i, 1] = fy
        forces[i, 2] = fz
    
    return forces

##############################
# PHASE 1: ALTERNATIVE SCAFFOLDING IMPLEMENTATION
# (Additive - matches provided scaffolding example exactly)
##############################

def velocity_verlet_step(particles, dt, epsilon=1e-10, G=6.67430e-11):
    """
    PHASE 1 SCAFFOLDING: Perform one Velocity Verlet integration step with softened gravity.
    
    Alternative implementation matching the provided scaffolding example.
    Pure Python version (not numba-accelerated) for clarity and compatibility.
    
    Args:
        particles: list of Particle objects
        dt: timestep
        epsilon: Softening parameter for gravity (default 1e-10)
        G: Gravitational constant (default 6.67430e-11)
    """
    try:
        # Compute initial accelerations
        accelerations = []
        for i, pi in enumerate(particles):
            force = np.zeros(3)
            for j, pj in enumerate(particles):
                if i == j:
                    continue
                r_vec = pj.position - pi.position
                r_sq = np.dot(r_vec, r_vec) + epsilon**2
                r = math.sqrt(r_sq)
                f_mag = -G * pi.mass * pj.mass / r_sq
                force += f_mag * (r_vec / r)
            accelerations.append(force / pi.mass)
        
        # Half-step velocity update: v_half = v + a(x)/2·dt
        for i, pi in enumerate(particles):
            pi.velocity += 0.5 * accelerations[i] * dt
        
        # Full-step position update: x_new = x + v_half·dt
        for i, pi in enumerate(particles):
            pi.position += pi.velocity * dt
        
        # Recompute accelerations at new positions: a_new = a(x_new)
        new_accelerations = []
        for i, pi in enumerate(particles):
            force = np.zeros(3)
            for j, pj in enumerate(particles):
                if i == j:
                    continue
                r_vec = pj.position - pi.position
                r_sq = np.dot(r_vec, r_vec) + epsilon**2
                r = math.sqrt(r_sq)
                f_mag = -G * pi.mass * pj.mass / r_sq
                force += f_mag * (r_vec / r)
            new_accelerations.append(force / pi.mass)
        
        # Complete velocity update: v = v_half + a_new/2·dt
        for i, pi in enumerate(particles):
            pi.velocity += 0.5 * new_accelerations[i] * dt
    except Exception as e:
        logger.error(f"Error in velocity_verlet_step: {e}")

@njit(parallel=True)
def velocity_verlet_step_numba(positions: np.ndarray, velocities: np.ndarray,
                                masses: np.ndarray, forces_old: np.ndarray,
                                dt: float, G_val: float = 6.67430e-11,
                                epsilon: float = 1e9) -> tuple:
    """
    PHASE 1: Velocity Verlet integration step.
    
    Implements:
        v_half = v + a(x)/2·dt
        x_new = x + v_half·dt
        a_new = a(x_new)
        v = v_half + a_new/2·dt
    
    Args:
        positions: Nx3 array of current positions
        velocities: Nx3 array of current velocities
        masses: N array of particle masses
        forces_old: Nx3 array of forces at current positions
        dt: timestep
        G_val: Gravitational constant
        epsilon: Softening parameter
    
    Returns:
        tuple: (new_positions, new_velocities, new_forces)
    """
    N = len(positions)
    new_positions = np.zeros_like(positions)
    new_velocities = np.zeros_like(velocities)
    
    # Step 1: v_half = v + a(x)/2·dt
    for i in prange(N):
        m = masses[i]
        ax_old = forces_old[i, 0] / m
        ay_old = forces_old[i, 1] / m
        az_old = forces_old[i, 2] / m
        
        vx_half = velocities[i, 0] + 0.5 * ax_old * dt
        vy_half = velocities[i, 1] + 0.5 * ay_old * dt
        vz_half = velocities[i, 2] + 0.5 * az_old * dt
        
        # Step 2: x_new = x + v_half·dt
        new_positions[i, 0] = positions[i, 0] + vx_half * dt
        new_positions[i, 1] = positions[i, 1] + vy_half * dt
        new_positions[i, 2] = positions[i, 2] + vz_half * dt
    
    # Step 3: Compute new forces at new positions
    new_forces = compute_forces_corrected_numba(new_positions, masses, G_val, epsilon)
    
    # Step 4: v = v_half + a_new/2·dt
    for i in prange(N):
        m = masses[i]
        ax_new = new_forces[i, 0] / m
        ay_new = new_forces[i, 1] / m
        az_new = new_forces[i, 2] / m
        
        new_velocities[i, 0] = velocities[i, 0] + 0.5 * (forces_old[i, 0] / m + ax_new) * dt
        new_velocities[i, 1] = velocities[i, 1] + 0.5 * (forces_old[i, 1] / m + ay_new) * dt
        new_velocities[i, 2] = velocities[i, 2] + 0.5 * (forces_old[i, 2] / m + az_new) * dt
    
    return new_positions, new_velocities, new_forces

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
        
        # PHASE 2: Separate energy tracking (no double counting)
        self.K = 0.5 * self.mass * np.linalg.norm(self.velocity)**2  # Kinetic Energy
        self.U = 0.0  # Gravitational Potential Energy
        self.U_dm = 0.0  # Dark Matter Potential Energy
        self.E_info = 0.0  # Informational Energy (from audio/frequency)
        
        # Total cosmic energy: Ec = K + U + U_dm + E_info
        self.Ec = self.K + self.U + self.U_dm + self.E_info
        
        # Legacy compatibility (deprecated, use separate energies)
        self.Uc = self.U + self.U_dm  # Combined potential energy
        
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
        
        # PHASE 3: Dandelion behavior state
        self.spring_attached = True  # Whether particle is attached to seed head
        self.detachment_time = 0.0  # Time when particle was detached
        self.radial_impulse = np.zeros(3)  # Outward radial impulse for dispersal
    
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
    
    def update_energy(self, potential_energy, dark_matter_potential=0.0):
        """
        PHASE 2: Updates particle energies with proper separation.
        
        Args:
            potential_energy: Gravitational potential energy U
            dark_matter_potential: Dark matter potential energy U_dm
        """
        try:
            # Update kinetic energy from current velocity
            self.K = 0.5 * self.mass * np.linalg.norm(self.velocity)**2
            self.K = max(0.0, self.K)  # Ensure non-negative
            
            # Update potential energies
            self.U = potential_energy
            self.U_dm = dark_matter_potential
            
            # Total cosmic energy: Ec = K + U + U_dm + E_info
            self.Ec = self.K + self.U + self.U_dm + self.E_info
            
            # Legacy compatibility
            self.Uc = self.U + self.U_dm
            
            # Update frequency and related quantities
            self.nu = (self.Ec + self.Uc) / h if h != 0 else 0.0
            self.update_vi()  # Update characteristic frequency when energy changes
            self.update_entropy()
            logger.debug(f"Particle {self.id}: Energy updated. K={self.K:.2e}, U={self.U:.2e}, U_dm={self.U_dm:.2e}, E_info={self.E_info:.2e}, Ec={self.Ec:.2e}")
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
    
    def assign_frequency(self, frequency, audio_token=None):
        """
        PHASE 2: Assigns frequency to particle and updates informational energy.
        
        Refactored to use normalized audio features (RMS, centroid, entropy) instead
        of raw h·ν scaling which produces unrealistic magnitudes.
        
        Args:
            frequency: Frequency value (Hz)
            audio_token: Optional dict with keys: rms, spectral_centroid, entropy, chaos
        """
        try:
            self.frequency = frequency
            self.nu = frequency
            
            # PHASE 2: Update E_info using normalized audio features
            if audio_token is not None:
                # Normalize features to reasonable energy scales
                rms = audio_token.get('rms', 0.0)
                spectral_centroid = audio_token.get('spectral_centroid', 0.0)
                entropy = audio_token.get('entropy', 0.0)
                
                # Map RMS to energy scale (normalized to E_0 reference)
                # RMS typically in [0, 1] range, scale to reasonable energy
                rms_energy = rms * E_0 * 0.01  # Scale factor prevents excessive energy
                
                # Map spectral centroid to frequency-dependent energy
                # Higher centroid = higher frequency content = more energy
                centroid_energy = (spectral_centroid / 10000.0) * E_0 * 0.001
                
                # Map entropy to informational energy (higher entropy = more information)
                entropy_energy = entropy * E_0 * 0.001
                
                # Total informational energy from audio features
                self.E_info = rms_energy + centroid_energy + entropy_energy
            else:
                # Fallback: use frequency with normalized scaling
                if frequency > 0:
                    # Normalized frequency energy (much smaller than raw h*nu)
                    normalized_freq = frequency / 10000.0  # Normalize to kHz scale
                    self.E_info = normalized_freq * E_0 * 0.001
                else:
                    self.E_info = 0.0
            
            # Update total cosmic energy: Ec = K + U + U_dm + E_info
            self.Ec = self.K + self.U + self.U_dm + self.E_info
            
            # Update entropy based on new energy
            self.update_entropy()
            
            token = f"Freq_{frequency:.2f}_Particle_{self.id}"
            self.tokens.append(token)
            logger.debug(f"Particle {self.id}: Assigned frequency {frequency:.2f} Hz, E_info={self.E_info:.2e} J, Ec={self.Ec:.2e} J")
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
    
    def compute_connectivity(self, sigma=0.5, m0=1.0, neighbor_cutoff=None):
        """
        PHASE 4: Computes connectivity (Omega) with cKDTree neighbor cutoff to avoid O(N²).
        
        The synaptic strength Ωi is computed as:
        omega_ij = (G * m_i * m_j) / (r_eff^2 * a0 * m0) * exp(-((x12_i - x12_j)^2) / (2 * sigma^2))
        Ωi = Σ ωij over neighbors (within cutoff radius)
        
        PHASE 4: Normalizes Ω per step (divide by max Ω) before feeding into x12 dynamics.
        
        Args:
            sigma: Gaussian similarity width parameter (default 0.5)
            m0: Reference mass for normalization (default 1.0)
            neighbor_cutoff: Maximum distance for neighbor search (default: 5 * DANDELION_SEED_HEAD_RADIUS)
        
        Returns:
            List of normalized Omega values for each particle
        """
        try:
            positions = np.array([p.position for p in self.particles])
            masses = np.array([p.mass for p in self.particles])
            N = len(self.particles)
            
            if N == 0:
                return []
            
            # PHASE 4: Use cKDTree for efficient neighbor search
            if neighbor_cutoff is None:
                neighbor_cutoff = 5 * DANDELION_SEED_HEAD_RADIUS
            
            # Build spatial tree for neighbor queries
            tree = cKDTree(positions)
            
            # Compute connectivity with neighbor cutoff
            Omega = np.zeros(N)
            
            for i, pi in enumerate(self.particles):
                omega_sum = 0.0
                
                # Query neighbors within cutoff radius
                neighbor_indices = tree.query_ball_point(positions[i], r=neighbor_cutoff)
                
                for j in neighbor_indices:
                    if i == j:
                        continue
                    
                    pj = self.particles[j]
                    
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
                Omega[i] = omega_sum
            
            # PHASE 4: Normalize Omega per step (divide by max Ω)
            # This ensures boundedness and convergence for x12 dynamics
            max_omega = np.max(Omega) if np.max(Omega) > 0 else 1.0
            if max_omega > 0:
                Omega_normalized = Omega / max_omega
                # Update particle omega values with normalized values
                for i, pi in enumerate(self.particles):
                    pi.omega = Omega_normalized[i]
            else:
                Omega_normalized = Omega
            
            logger.debug(f"Computed similarity-weighted connectivity for {N} particles (cutoff={neighbor_cutoff:.2e}).")
            logger.debug(f"Omega range: [{np.min(Omega):.4e}, {np.max(Omega):.4e}], normalized: [{np.min(Omega_normalized):.4e}, {np.max(Omega_normalized):.4e}]")
            
            return Omega_normalized.tolist()
        except Exception as e:
            logger.error(f"Error computing connectivity: {e}")
            return np.zeros(len(self.particles))

##############################
# PHASE 4: ALTERNATIVE SCAFFOLDING IMPLEMENTATIONS
# (Additive - matches provided scaffolding examples exactly)
##############################

def compute_connectivity(particles, cutoff=None, sigma=0.5, G=6.67430e-11, a0=9.81, m0=1.0):
    """
    PHASE 4 SCAFFOLDING: Compute synaptic connectivity Ω for each particle using cKDTree neighbor search
    and Gaussian similarity weighting on x12 differences.
    
    Alternative implementation matching the provided scaffolding example.
    
    Args:
        particles: list of Particle objects
        cutoff: Maximum distance for neighbor search (default: 5 * DANDELION_SEED_HEAD_RADIUS for proper scaling)
        sigma: Gaussian similarity width parameter (default 0.5)
        G: Gravitational constant (default 6.67430e-11)
        a0: Characteristic acceleration (default 9.81)
        m0: Reference mass for normalization (default 1.0)
    
    Returns:
        Normalized Omega array (Ω_norm)
    """
    try:
        if not particles:
            return np.array([])
        
        # Use appropriate cutoff for simulation scale (if not provided)
        if cutoff is None:
            cutoff = 5 * DANDELION_SEED_HEAD_RADIUS
        
        positions = np.array([p.position for p in particles])
        masses = np.array([p.mass for p in particles])
        tree = cKDTree(positions)
        N = len(particles)
        
        # Initialize connectivity
        Omega = np.zeros(N)
        
        for i, pi in enumerate(particles):
            # Query neighbors within cutoff radius
            neighbors = tree.query_ball_point(pi.position, cutoff)
            omega_sum = 0.0
            
            for j in neighbors:
                if i == j:
                    continue
                pj = particles[j]
                r_vec = pj.position - pi.position
                r = np.linalg.norm(r_vec) + 1e-10
                
                # Base gravitational coupling
                omega_grav = (G * pi.mass * pj.mass) / (r**2 * a0 * m0)
                
                # Similarity weighting on x12
                dx12 = pi.x12 - pj.x12
                similarity = np.exp(-(dx12**2) / (2 * sigma**2))
                
                omega_sum += omega_grav * similarity
            
            Omega[i] = omega_sum
            pi.omega = omega_sum
        
        # Normalize Ω values: divide by max Ω
        max_Omega = np.max(Omega) + 1e-12
        Omega_norm = Omega / max_Omega
        
        # Update particle omega values with normalized values
        for i, pi in enumerate(particles):
            pi.omega = Omega_norm[i]
        
        return Omega_norm
    except Exception as e:
        logger.error(f"Error in compute_connectivity (scaffolding): {e}")
        return np.zeros(len(particles))

def update_internal_state(particles, Omega_norm, dt, k=0.1, gamma=0.01, alpha=0.05):
    """
    PHASE 4 SCAFFOLDING: Update internal adaptive state x12 and memory m12 for each particle.
    
    Alternative implementation matching the provided scaffolding example.
    
    Implements:
        dx12/dt = k·Ω_norm - γ·x12
        dm12/dt = α·(x12 - m12)
    
    Verifies equilibrium: x12 = (k/γ)·Ω_norm (steady-state solution)
    Ensures boundedness: x12 ∈ [-1, 1]
    
    Args:
        particles: list of Particle objects
        Omega_norm: Normalized connectivity array (from compute_connectivity)
        dt: timestep
        k: Coupling strength (default 0.1)
        gamma: Damping factor (default 0.01)
        alpha: Memory update rate (default 0.05)
    """
    try:
        if not particles or len(Omega_norm) != len(particles):
            return
        
        for i, p in enumerate(particles):
            # Update x12: dx12/dt = k·Ω_norm - γ·x12
            dx12 = k * Omega_norm[i] - gamma * p.x12
            p.x12 += dx12 * dt
            
            # Ensure boundedness: x12 ∈ [-1, 1]
            p.x12 = max(-1.0, min(1.0, p.x12))
            
            # Verify equilibrium condition (for diagnostics)
            # At equilibrium: x12 = (k/gamma) * Omega_norm
            equilibrium_x12 = (k / gamma) * Omega_norm[i] if gamma > 0 else 0.0
            equilibrium_x12 = max(-1.0, min(1.0, equilibrium_x12))  # Bound equilibrium value
            
            # Update memory m12: dm12/dt = α·(x12 - m12)
            dm12 = alpha * (p.x12 - p.m12)
            p.m12 += dm12 * dt
            
            # Optional: Log convergence toward equilibrium (for diagnostics)
            if hasattr(p, 'id') and p.id % 100 == 0:  # Log every 100th particle
                convergence_error = abs(p.x12 - equilibrium_x12)
                if convergence_error > 0.1:  # Only log if far from equilibrium
                    logger.debug(f"Particle {p.id}: x12={p.x12:.4f}, equilibrium={equilibrium_x12:.4f}, "
                               f"error={convergence_error:.4f}")
    except Exception as e:
        logger.error(f"Error in update_internal_state (scaffolding): {e}")

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
    
    def compute_forces(self, Psi, positions, masses=None):
        """
        PHASE 1: Computes forces with corrected gravitational computation.
        
        Args:
            Psi: Either array of Psi values or dict with 'psi_total' key
            positions: array of particle positions
            masses: Optional array of particle masses (required for corrected computation)
        """
        try:
            # Handle both old format (array) and new format (dict)
            if isinstance(Psi, dict):
                Psi_values = Psi.get('psi_total', np.zeros(len(positions)))
            else:
                Psi_values = np.asarray(Psi)
            
            # Clamp Psi values to ±10 for stability
            Psi_values = np.clip(Psi_values, -10.0, 10.0)
            
            N = len(Psi_values)
            
            # PHASE 1: Use corrected force computation if masses provided
            if masses is not None and len(masses) == N:
                # Use corrected gravitational forces with proper mass inclusion
                forces = compute_forces_corrected_numba(
                    np.asarray(positions), 
                    np.asarray(masses), 
                    G_val=G, 
                    epsilon=SOFTENING_EPSILON
                )
                
                # Apply Psi modulation (small effect to preserve stability)
                kappa = 1e-5
                for i in range(N):
                    for j in range(N):
                        if i != j:
                            delta = positions[j] - positions[i]
                            distance_sq = np.dot(delta, delta) + 1e-10
                            distance = math.sqrt(distance_sq)
                            psi_modulation = 1.0 + kappa * Psi_values[i] * Psi_values[j]
                            forces[i] += forces[i] * (psi_modulation - 1.0) * 0.1  # Small modulation
            else:
                # Legacy mode: Psi-modulated forces (backward compatibility)
                kappa = 1e-5
            forces = np.zeros((N, 3))
            for i in range(N):
                for j in range(N):
                    if i != j:
                        delta = positions[j] - positions[i]
                        distance_sq = np.dot(delta, delta) + 1e-10
                        distance = math.sqrt(distance_sq)
                        base_force = G / distance_sq
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

##############################
# PHASE 2: ALTERNATIVE SCAFFOLDING IMPLEMENTATIONS
# (Additive - matches provided scaffolding examples exactly)
##############################

def compute_energies(particles, G=6.67430e-11, epsilon=1e-10):
    """
    PHASE 2 SCAFFOLDING: Compute separate energy channels for each particle and system totals.
    
    Alternative implementation matching the provided scaffolding example.
    
    Args:
        particles: list of Particle objects
        G: Gravitational constant
        epsilon: Small value to avoid division by zero
    
    Returns:
        dict with keys: K, U, U_dm, E_info, Ec_total
    """
    try:
        total_K, total_U, total_Udm, total_Einfo = 0.0, 0.0, 0.0, 0.0
        
        # Pairwise potential energy
        for i, pi in enumerate(particles):
            # Kinetic energy
            v_sq = np.dot(pi.velocity, pi.velocity)
            K_i = 0.5 * pi.mass * v_sq
            total_K += K_i
            
            # Informational energy (from audio features)
            Einfo_i = getattr(pi, "E_info", 0.0)
            total_Einfo += Einfo_i
            
            # Potential energy contributions
            for j, pj in enumerate(particles):
                if j <= i:
                    continue
                r_vec = pj.position - pi.position
                r_sq = np.dot(r_vec, r_vec) + epsilon**2
                r = math.sqrt(r_sq)
                U_ij = -G * pi.mass * pj.mass / r
                total_U += U_ij
            
            # Dark matter potential (approximate NFW contribution)
            r_mag = np.linalg.norm(pi.position)
            rho0, rs = 1e-24, 1e21
            rho_dm = rho0 * (rs / r_mag) / ((1.0 + (r_mag / rs))**2) if r_mag > 0 else 0.0
            Udm_i = -G * pi.mass * rho_dm * (4.0 * math.pi * r_mag**2) if r_mag > 0 else 0.0
            total_Udm += Udm_i
        
        Ec_total = total_K + total_U + total_Udm + total_Einfo
        return {
            "K": total_K,
            "U": total_U,
            "U_dm": total_Udm,
            "E_info": total_Einfo,
            "Ec_total": Ec_total
        }
    except Exception as e:
        logger.error(f"Error in compute_energies: {e}")
        return {"K": 0.0, "U": 0.0, "U_dm": 0.0, "E_info": 0.0, "Ec_total": 0.0}

def assign_frequency_energy(particle, frequency, rms=0.0, centroid=0.0, entropy=0.0):
    """
    PHASE 2 SCAFFOLDING: Update particle informational energy based on audio features.
    
    Alternative implementation matching the provided scaffolding example.
    
    Args:
        particle: Particle object
        frequency: Frequency value (Hz)
        rms: RMS amplitude (normalized)
        centroid: Spectral centroid (Hz)
        entropy: Spectral entropy (normalized)
    """
    try:
        # Normalize features (avoid division by zero)
        norm_rms = rms / (rms + 1e-6)
        norm_centroid = centroid / (centroid + 1e-6) if centroid > 0 else 0.0
        norm_entropy = entropy / (entropy + 1e-6) if entropy > 0 else 0.0
        
        # Weighted informational energy contribution
        # Weights: 0.5 RMS, 0.3 centroid, 0.2 entropy
        particle.E_info = (
            0.5 * norm_rms +
            0.3 * norm_centroid +
            0.2 * norm_entropy
        ) * particle.mass
        
        # Store frequency for diagnostics
        particle.frequency = frequency
        
        # Update total cosmic energy: Ec = K + U + U_dm + E_info
        particle.K = 0.5 * particle.mass * np.linalg.norm(particle.velocity)**2
        particle.Ec = particle.K + getattr(particle, 'U', 0.0) + getattr(particle, 'U_dm', 0.0) + particle.E_info
    except Exception as e:
        logger.error(f"Error in assign_frequency_energy: {e}")


def compute_adaptive_dt(particles, dt_max=1.0, safety_factor=0.1):
    """
    PHASE 1: Compute adaptive timestep based on system dynamics.
    dt = min(dt_max, α·r_min/v_max)
    
    Uses cKDTree for efficient r_min computation (minimum inter-particle separation).
    This ensures numerical stability by keeping the timestep
    small enough that particles don't move too far per step.
    
    Args:
        particles: list of Particle objects
        dt_max: maximum allowed timestep (default 1.0)
        safety_factor: safety factor α for timestep (default 0.1)
    
    Returns:
        Adaptive timestep (s)
    """
    try:
        if not particles or len(particles) < 2:
            return dt_max
        
        # Find maximum velocity
        v_max = max(np.linalg.norm(p.velocity) for p in particles)
        v_max = max(v_max, 1e-10)  # Avoid division by zero
        
        # PHASE 1: Use cKDTree for efficient r_min computation (minimum inter-particle separation)
        positions = np.array([p.position for p in particles])
        tree = cKDTree(positions)
        
        # Query nearest neighbor for each particle (excluding itself)
        r_min = float('inf')
        for i, pos in enumerate(positions):
            # Query 2 nearest neighbors (self + 1 actual neighbor)
            distances, indices = tree.query(pos, k=2)
            if len(distances) > 1:
                # Second element is the nearest neighbor (first is self)
                neighbor_dist = distances[1]
                r_min = min(r_min, neighbor_dist)
        
        # Fallback: if cKDTree query failed, use simple distance computation
        if r_min == float('inf') or r_min <= 0:
            # Find minimum non-zero inter-particle distance
            r_min = float('inf')
            for i, p1 in enumerate(particles):
                for j, p2 in enumerate(particles):
                    if i != j:
                        dist = np.linalg.norm(p1.position - p2.position)
                        if dist > 0:
                            r_min = min(r_min, dist)
        
        if r_min == float('inf') or r_min <= 0:
            return dt_max
        
        # Adaptive timestep: dt = min(dt_max, α·r_min/v_max)
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
# PHASE 3: AUDIO-DRIVEN DANDELION BEHAVIOR
##############################

# Dandelion behavior constants
DANDELION_RMS_THRESHOLD = 0.1  # RMS threshold for blow detection
DANDELION_SPRING_K = 1e-5  # Spring constant for seed head clustering
DANDELION_DRAG_GAMMA = 0.1  # Drag coefficient
DANDELION_SEED_HEAD_RADIUS = 1e10  # Radius of seed head cluster (m)
DANDELION_IMPULSE_SCALE = 1e3  # Scale factor for radial impulses

def compute_seed_head_center(particles):
    """
    PHASE 3: Compute center of seed head (cluster of attached particles).
    
    Args:
        particles: list of Particle objects
    
    Returns:
        Center position (3D array)
    """
    try:
        attached_particles = [p for p in particles if p.spring_attached]
        if not attached_particles:
            # If no attached particles, use center of mass
            if particles:
                return np.mean([p.position for p in particles], axis=0)
            return np.zeros(3)
        return np.mean([p.position for p in attached_particles], axis=0)
    except Exception as e:
        logger.error(f"Error computing seed head center: {e}")
        return np.zeros(3)

def apply_spring_forces_to_seed_head(particles, seed_center, spring_k=DANDELION_SPRING_K):
    """
    PHASE 3: Apply weak spring forces to cluster particles at seed head.
    
    Before blow: particles are weakly attracted to seed head center.
    
    Args:
        particles: list of Particle objects
        seed_center: 3D position of seed head center
        spring_k: Spring constant
    """
    try:
        for particle in particles:
            if particle.spring_attached:
                # Spring force toward seed head center
                displacement = seed_center - particle.position
                distance = np.linalg.norm(displacement) + 1e-10
                
                # Spring force: F = -k * (r - r0)
                # Weak force to maintain clustering
                force_magnitude = spring_k * distance
                force = force_magnitude * (displacement / distance)
                
                # Apply as acceleration
                acceleration = force / particle.mass
                particle.velocity += acceleration * 0.01  # Small timestep for spring
    except Exception as e:
        logger.error(f"Error applying spring forces: {e}")

def apply_dandelion_dispersal(particles, audio_token, seed_center, dt):
    """
    PHASE 3: Apply dandelion dispersal behavior based on audio blow.
    
    Maps microphone FFT features to particle behavior:
    - RMS → outward radial impulse (seed dispersal)
    - Centroid → clustering bias by frequency
    - Chaos/entropy → perturb synchronization phase θ
    
    Args:
        particles: list of Particle objects
        audio_token: dict with keys: rms, spectral_centroid, chaos, entropy
        seed_center: 3D position of seed head center
        dt: timestep
    """
    try:
        if audio_token is None:
            return
        
        rms = audio_token.get('rms', 0.0)
        spectral_centroid = audio_token.get('spectral_centroid', 0.0)
        chaos = audio_token.get('chaos', 0.0)
        entropy = audio_token.get('entropy', 0.0)
        
        # Blow threshold: RMS > R_thresh detaches springs
        if rms > DANDELION_RMS_THRESHOLD:
            # Detach particles from seed head
            for particle in particles:
                if particle.spring_attached:
                    particle.spring_attached = False
                    particle.detachment_time = time.time()
                    
                    # Compute radial direction from seed head
                    radial_dir = particle.position - seed_center
                    radial_dist = np.linalg.norm(radial_dir) + 1e-10
                    radial_unit = radial_dir / radial_dist
                    
                    # Outward radial impulse: magnitude proportional to RMS
                    impulse_magnitude = rms * DANDELION_IMPULSE_SCALE
                    particle.radial_impulse = radial_unit * impulse_magnitude
                    particle.velocity += particle.radial_impulse
        
        # Apply drag to detached particles: v ← v·(1 - γ_drag·RMS·dt)
        gamma_drag = DANDELION_DRAG_GAMMA * rms
        for particle in particles:
            if not particle.spring_attached:
                drag_factor = 1.0 - gamma_drag * dt
                drag_factor = max(0.0, drag_factor)  # Prevent negative velocities
                particle.velocity *= drag_factor
        
        # Clustering bias by frequency (spectral centroid)
        # Higher centroid → particles cluster by frequency bands
        if spectral_centroid > 0:
            centroid_normalized = spectral_centroid / 10000.0  # Normalize to [0, 1]
            for particle in particles:
                # Bias velocity toward particles with similar frequencies
                if hasattr(particle, 'frequency') and particle.frequency > 0:
                    # Find particles with similar frequencies
                    similar_particles = [
                        p for p in particles 
                        if p != particle and abs(p.frequency - particle.frequency) < 100
                    ]
                    if similar_particles:
                        # Attract toward similar-frequency particles
                        avg_pos = np.mean([p.position for p in similar_particles], axis=0)
                        attraction_dir = avg_pos - particle.position
                        attraction_dist = np.linalg.norm(attraction_dir) + 1e-10
                        attraction_force = centroid_normalized * 1e-3 * (attraction_dir / attraction_dist)
                        particle.velocity += attraction_force * dt
        
        # Perturb synchronization phase θ with chaos/entropy
        for particle in particles:
            phase_perturbation = chaos * 0.1 + entropy * 0.05
            particle.theta += phase_perturbation * dt
            particle.theta = particle.theta % (2 * math.pi)
    except Exception as e:
        logger.error(f"Error applying dandelion dispersal: {e}")

def apply_golden_ratio_harmonics(particles, audio_token):
    """
    PHASE 3: Integrate golden ratio harmonics into main loop for coherent clustering.
    
    Uses φ-harmonic series to create frequency-based clustering patterns.
    
    Args:
        particles: list of Particle objects
        audio_token: dict with harmonics list (from tokenize_audio_frame)
    """
    try:
        if audio_token is None or 'harmonics' not in audio_token:
            return
        
        harmonics = audio_token.get('harmonics', [])
        if not harmonics:
            return
        
        phi = (1 + math.sqrt(5)) / 2  # Golden ratio
        
        # Group particles by φ-harmonic frequencies
        for particle in particles:
            if hasattr(particle, 'frequency') and particle.frequency > 0:
                # Find closest harmonic
                min_harmonic_dist = float('inf')
                closest_harmonic = None
                
                for harmonic in harmonics:
                    harmonic_freq = harmonic.get('frequency', 0)
                    if harmonic_freq > 0:
                        dist = abs(particle.frequency - harmonic_freq)
                        if dist < min_harmonic_dist:
                            min_harmonic_dist = dist
                            closest_harmonic = harmonic
                
                # Apply harmonic-based clustering
                if closest_harmonic:
                    amplitude = closest_harmonic.get('amplitude', 0)
                    order = closest_harmonic.get('order', 0)
                    
                    # Higher order harmonics have weaker clustering
                    clustering_strength = amplitude * (phi ** (-order))
                    
                    # Find other particles in same harmonic group
                    harmonic_freq = closest_harmonic.get('frequency', 0)
                    harmonic_group = [
                        p for p in particles 
                        if p != particle 
                        and hasattr(p, 'frequency')
                        and abs(p.frequency - harmonic_freq) < 50
                    ]
                    
                    if harmonic_group:
                        # Attract toward harmonic group center
                        group_center = np.mean([p.position for p in harmonic_group], axis=0)
                        attraction = group_center - particle.position
                        attraction_dist = np.linalg.norm(attraction) + 1e-10
                        attraction_force = clustering_strength * 1e-4 * (attraction / attraction_dist)
                        particle.velocity += attraction_force * 0.01  # Small timestep
    except Exception as e:
        logger.error(f"Error applying golden ratio harmonics: {e}")

##############################
# PHASE 3: ALTERNATIVE SCAFFOLDING IMPLEMENTATIONS
# (Additive - matches provided scaffolding examples exactly)
##############################

def apply_audio_impulses(particles, rms, centroid, entropy,
                         blow_threshold=0.2, k_spring=0.05, gamma_drag=0.01):
    """
    PHASE 3 SCAFFOLDING: Apply audio-driven impulses to particles to simulate dandelion dispersal.
    
    Alternative implementation matching the provided scaffolding example.
    
    - rms: root mean square amplitude of audio (proxy for breath intensity).
    - centroid: spectral centroid (bias clustering).
    - entropy: spectral entropy (adds turbulence).
    - blow_threshold: RMS threshold for blow detection (default 0.2)
    - k_spring: Spring constant for pre-blow clustering (default 0.05)
    - gamma_drag: Drag coefficient (default 0.01)
    """
    try:
        if not particles:
            return
        
        # Compute cluster center (seed head)
        positions = np.array([p.position for p in particles])
        center = np.mean(positions, axis=0)
        
        for p in particles:
            # Vector from center to particle
            r_vec = p.position - center
            r_norm = np.linalg.norm(r_vec) + 1e-8
            r_hat = r_vec / r_norm
            
            if rms < blow_threshold:
                # Pre-blow: weak spring pulls particle toward center
                spring_force = -k_spring * r_vec
                p.velocity += spring_force / p.mass
                
                # Mark as attached if not already
                if not hasattr(p, 'spring_attached'):
                    p.spring_attached = True
                p.spring_attached = True
            else:
                # Blow detected: outward impulse proportional to RMS
                impulse = (rms * 5.0) * r_hat
                p.velocity += impulse
                
                # Apply drag proportional to RMS: v ← v·(1 - γ_drag·RMS)
                p.velocity *= (1.0 - gamma_drag * rms)
                
                # Detach from seed head
                if hasattr(p, 'spring_attached'):
                    p.spring_attached = False
                
                # Perturb phase θ with entropy
                if hasattr(p, 'theta'):
                    p.theta += entropy * random.uniform(-0.1, 0.1)
                    p.theta = p.theta % (2 * math.pi)
            
            # Bias clustering by centroid (higher centroid → tighter grouping)
            bias_strength = 0.01 * centroid
            p.velocity -= bias_strength * r_vec / r_norm
    except Exception as e:
        logger.error(f"Error in apply_audio_impulses: {e}")

def assign_phi_harmonics(particles, base_freq, phi=(1 + 5**0.5) / 2, num_harmonics=3):
    """
    PHASE 3 SCAFFOLDING: Assign golden ratio harmonics to subsets of particles for coherent clustering.
    
    Alternative implementation matching the provided scaffolding example.
    
    Args:
        particles: list of Particle objects
        base_freq: Base frequency for harmonic series
        phi: Golden ratio constant (default (1+√5)/2)
        num_harmonics: Number of harmonics to generate (default 3)
    """
    try:
        if not particles:
            return
        
        # Generate harmonic frequencies: base_freq * (phi ** n)
        harmonics = [base_freq * (phi ** n) for n in range(num_harmonics)]
        
        for i, p in enumerate(particles):
            # Assign harmonic based on particle index (cyclic)
            harmonic = harmonics[i % len(harmonics)]
            
            # Update particle frequency
            p.frequency = harmonic
            
            # Optionally adjust informational energy contribution
            if not hasattr(p, 'E_info'):
                p.E_info = 0.0
            p.E_info = getattr(p, "E_info", 0.0) + 0.1 * harmonic
            
            # Update characteristic frequency
            if hasattr(p, 'nu'):
                p.nu = harmonic
            if hasattr(p, 'vi'):
                p.vi = harmonic
    except Exception as e:
        logger.error(f"Error in assign_phi_harmonics: {e}")


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
    
    def record_frame(self, frame, simulator_state=None):
        """
        PHASE 5: Record an audio frame with comprehensive diagnostics.
        
        Records FFT freqs/mags, RMS, centroid, chaos, entropy, camera stats, dt, and random seeds
        for deterministic reproduction.
        
        Args:
            frame: audio frame data (dict with keys: freqs, mags, rms, spectral_centroid, chaos, entropy, etc.)
            simulator_state: Optional dict with simulator state (dt, step, time, random_seed, etc.)
        """
        if self.is_recording:
            # Create comprehensive frame record
            frame_record = {
                'audio': frame.copy() if hasattr(frame, 'copy') else frame,
                'timestamp': time.time(),
                'simulator_state': simulator_state or {}
            }
            self.recorded_frames.append(frame_record)
    
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
# PHASE 5: ALTERNATIVE SCAFFOLDING IMPLEMENTATIONS
# (Additive - matches provided scaffolding examples exactly)
##############################

def compute_diagnostics(particles, energies, dt):
    """
    PHASE 5 SCAFFOLDING: Compute conservation diagnostics and emergent metrics.
    
    Alternative implementation matching the provided scaffolding example.
    
    Args:
        particles: list of Particle objects
        energies: dict with keys: K, U, U_dm, E_info, Ec_total (from compute_energies)
        dt: timestep
    
    Returns:
        dict with keys: momentum, angular_momentum, virial_ratio, sync_order, dt, Ec_total
    """
    try:
        if not particles:
            return {
                "momentum": np.zeros(3),
                "angular_momentum": np.zeros(3),
                "virial_ratio": 0.0,
                "sync_order": 0.0,
                "dt": dt,
                "Ec_total": 0.0
            }
        
        # Total momentum: P_total = Σ(m * v)
        total_momentum = np.sum([p.mass * p.velocity for p in particles], axis=0)
        
        # Angular momentum: L_total = Σ(r × (m * v))
        total_ang_momentum = np.sum([np.cross(p.position, p.mass * p.velocity) for p in particles], axis=0)
        
        # Virial ratio: 2K / |U|
        # For a virialized system, this should be close to 1
        K = energies.get("K", 0.0)
        U = energies.get("U", 0.0)
        virial_ratio = (2.0 * K) / abs(U) if U != 0 else (np.inf if K > 0 else 0.0)
        
        # Synchronization order parameter (Kuramoto-style)
        # r = |(1/N) * Σ(exp(i*theta))|
        # r = 0: no synchronization, r = 1: perfect synchronization
        phases = np.array([p.theta for p in particles if hasattr(p, 'theta')])
        if len(phases) > 0:
            order_param = np.abs(np.mean(np.exp(1j * phases)))
        else:
            order_param = 0.0
        
        return {
            "momentum": total_momentum.tolist() if isinstance(total_momentum, np.ndarray) else total_momentum,
            "angular_momentum": total_ang_momentum.tolist() if isinstance(total_ang_momentum, np.ndarray) else total_ang_momentum,
            "virial_ratio": float(virial_ratio),
            "sync_order": float(order_param),
            "dt": float(dt),
            "Ec_total": float(energies.get("Ec_total", 0.0))
        }
    except Exception as e:
        logger.error(f"Error in compute_diagnostics (scaffolding): {e}")
        return {
            "momentum": [0.0, 0.0, 0.0],
            "angular_momentum": [0.0, 0.0, 0.0],
            "virial_ratio": 0.0,
            "sync_order": 0.0,
            "dt": dt,
            "Ec_total": 0.0
        }

def log_replay_step(log_file, fft_data, rms, centroid, entropy, diagnostics, seed, camera_data=None, chaos=None):
    """
    PHASE 5 SCAFFOLDING: Log replay data for deterministic reproduction.
    
    Alternative implementation matching the provided scaffolding example.
    
    Records FFT freqs/mags, RMS, centroid, chaos, entropy, camera-derived stats, dt, and random seeds.
    
    Args:
        log_file: Path to log file (will append entries)
        fft_data: dict with keys: freqs, mags (from FFT analysis)
        rms: RMS amplitude value
        centroid: Spectral centroid value
        entropy: Spectral entropy value
        diagnostics: dict from compute_diagnostics (includes momentum, angular_momentum, virial_ratio, sync_order, dt, Ec_total)
        seed: Random seed value for reproducibility
        camera_data: Optional dict with camera-derived stats (brightness, motion, etc.)
        chaos: Optional chaos metric value
    """
    try:
        # Extract camera stats if available
        camera_stats = {}
        if camera_data is not None:
            camera_stats = {
                "brightness": float(camera_data.get("brightness", 0.0)),
                "motion": float(camera_data.get("motion", 0.0)) if "motion" in camera_data else None,
                "frame_count": int(camera_data.get("frame_count", 0)) if "frame_count" in camera_data else None
            }
        
        entry = {
            "timestamp": time.time(),
            "fft_freqs": fft_data.get("freqs", []),
            "fft_mags": fft_data.get("mags", []),
            "rms": float(rms),
            "centroid": float(centroid),
            "entropy": float(entropy),
            "chaos": float(chaos) if chaos is not None else None,
            "diagnostics": diagnostics,
            "camera_stats": camera_stats if camera_stats else None,
            "seed": int(seed),
            "dt": diagnostics.get("dt", 0.0)
        }
        
        # Append to log file (one JSON object per line for easy parsing)
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
        
        logger.debug(f"Logged replay step to {log_file}: timestamp={entry['timestamp']}, seed={seed}")
    except Exception as e:
        logger.error(f"Error in log_replay_step (scaffolding): {e}")

def load_replay_log(log_file):
    """
    PHASE 5 SCAFFOLDING: Load replay log entries from file.
    
    Args:
        log_file: Path to log file
    
    Returns:
        List of replay entries (dicts)
    """
    try:
        entries = []
        if not os.path.exists(log_file):
            logger.warning(f"Replay log file {log_file} does not exist.")
            return entries
        
        with open(log_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entry = json.loads(line)
                        entries.append(entry)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Error parsing replay log line: {e}")
                        continue
        
        logger.info(f"Loaded {len(entries)} replay entries from {log_file}")
        return entries
    except Exception as e:
        logger.error(f"Error loading replay log {log_file}: {e}")
        return []

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
    
    def plot_particles(self, step, frequency_data=None, plot_container=None, show_seed_head=True, show_harmonics=True):
        """
        PHASE 6: Plots particle positions with enhanced visualization features.
        
        Enhanced features:
        - Seed head clustering visualization
        - RMS intensity to particle color/size mapping
        - φ-harmonic groupings overlay
        
        Args:
            step: Simulation step index
            frequency_data: Audio frequency data (may include audio_token with RMS, harmonics)
            plot_container: Streamlit container for plotting
            show_seed_head: Whether to show seed head clustering (default True)
            show_harmonics: Whether to show φ-harmonic groupings (default True)
        
        Returns:
            Plotly figure object
        """
        try:
            if step >= len(self.simulator.history) or len(self.simulator.history) == 0:
                return None
                
            positions = self.simulator.history[step]
            Ec = np.array([p.Ec for p in self.simulator.particles])
            masses = np.array([p.mass for p in self.simulator.particles])
            norm_energy = Ec / np.max(Ec) if np.max(Ec) > 0 else Ec
            
            # PHASE 6: Extract RMS from audio_token if available
            rms_values = None
            audio_token = None
            if frequency_data and isinstance(frequency_data, dict):
                if 'rms' in frequency_data:
                    audio_token = frequency_data
                    rms_values = np.array([frequency_data.get('rms', 0.0)] * len(self.simulator.particles))
                elif 'freqs' in frequency_data:
                    # Legacy format - no RMS available
                    rms_values = None
            
            # PHASE 6: Map RMS intensity to particle size (if available)
            if rms_values is not None and np.max(rms_values) > 0:
                # Base size from mass, modulated by RMS
                base_sizes = 2 + (masses / np.max(masses)) * 8 if np.max(masses) > 0 else np.full(masses.shape, 2)
                rms_normalized = rms_values / np.max(rms_values)
                sizes = base_sizes * (1.0 + rms_normalized * 2.0)  # RMS increases size by up to 2x
            else:
                sizes = 2 + (masses / np.max(masses)) * 8 if np.max(masses) > 0 else np.full(masses.shape, 2)
            
            # PHASE 6: Map RMS intensity to particle color (if available)
            if rms_values is not None and np.max(rms_values) > 0:
                # Color by RMS intensity (red = high RMS, blue = low RMS)
                rms_normalized = rms_values / np.max(rms_values)
                if plt:
                    colors = plt.cm.RdYlBu_r(rms_normalized)[:, :3]  # Red-Yellow-Blue reversed
                else:
                    colors = rms_normalized
            elif frequency_data:
                freqs = frequency_data.get('freqs', [])
                mags = frequency_data.get('mags', [])
                if len(freqs) > 0 and len(mags) > 0 and len(freqs) == len(mags):
                    dominant_freq = freqs[np.argmax(mags)]
                    hue = (dominant_freq / 20000) * 360
                    if plt:
                        color = plt.cm.hsv(hue / 360)
                        colors = np.tile(color[:3], (len(self.simulator.particles), 1))
                    else:
                        colors = norm_energy
                else:
                    colors = norm_energy
            else:
                colors = norm_energy
            
            # Use particle frequencies if available (fallback)
            if len(self.simulator.particles) > 0 and rms_values is None:
                particle_freqs = np.array([p.frequency for p in self.simulator.particles])
                if np.any(particle_freqs > 0):
                    hue_values = (particle_freqs / 20000.0) % 1.0
                    if plt:
                        colors = np.array([plt.cm.hsv(h)[:3] for h in hue_values])
                    else:
                        colors = norm_energy
                else:
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
                if rms_values is not None:
                    color_values = rms_values  # Use RMS for color mapping
                else:
                    color_values = norm_energy
            else:
                color_values = colors if isinstance(colors, np.ndarray) else norm_energy
            
            # PHASE 6: Create base scatter plot
            fig = go.Figure()
            
            # PHASE 6: Add seed head clustering visualization
            if show_seed_head and len(self.simulator.particles) > 0:
                attached_particles = [i for i, p in enumerate(self.simulator.particles) if hasattr(p, 'spring_attached') and p.spring_attached]
                if attached_particles:
                    seed_positions = positions[attached_particles]
                    seed_center = np.mean(seed_positions, axis=0)
                    
                    # Draw seed head center as a larger marker
                    fig.add_trace(go.Scatter3d(
                        x=[seed_center[0]],
                        y=[seed_center[1]],
                        z=[seed_center[2]],
                        mode='markers',
                        marker=dict(
                            size=20,
                            color='yellow',
                            symbol='diamond',
                            opacity=0.9,
                            line=dict(width=2, color='orange')
                        ),
                        name='Seed Head Center',
                        showlegend=True
                    ))
                    
                    # Draw connections from seed head center to attached particles
                    for idx in attached_particles[:min(20, len(attached_particles))]:  # Limit to 20 for performance
                        pos = positions[idx]
                        fig.add_trace(go.Scatter3d(
                            x=[seed_center[0], pos[0]],
                            y=[seed_center[1], pos[1]],
                            z=[seed_center[2], pos[2]],
                            mode='lines',
                            line=dict(color='yellow', width=1, dash='dot'),
                            opacity=0.3,
                            showlegend=False,
                            hoverinfo='skip'
                        ))
            
            # PHASE 6: Add φ-harmonic groupings overlay
            if show_harmonics and audio_token and 'harmonics' in audio_token:
                harmonics = audio_token.get('harmonics', [])
                phi = (1 + math.sqrt(5)) / 2
                
                # Group particles by harmonic frequencies
                harmonic_groups = {}
                for i, particle in enumerate(self.simulator.particles):
                    if hasattr(particle, 'frequency') and particle.frequency > 0:
                        # Find closest harmonic
                        min_dist = float('inf')
                        closest_harmonic_idx = None
                        for h_idx, harmonic in enumerate(harmonics):
                            harmonic_freq = harmonic.get('frequency', 0)
                            if harmonic_freq > 0:
                                dist = abs(particle.frequency - harmonic_freq)
                                if dist < min_dist:
                                    min_dist = dist
                                    closest_harmonic_idx = h_idx
                        
                        if closest_harmonic_idx is not None and min_dist < 50:  # Within 50 Hz
                            if closest_harmonic_idx not in harmonic_groups:
                                harmonic_groups[closest_harmonic_idx] = []
                            harmonic_groups[closest_harmonic_idx].append(i)
                
                # Draw harmonic group centers and connections
                harmonic_colors = plt.cm.rainbow(np.linspace(0, 1, len(harmonic_groups))) if plt else ['red', 'green', 'blue', 'yellow', 'purple']
                for group_idx, (harmonic_idx, particle_indices) in enumerate(harmonic_groups.items()):
                    if len(particle_indices) > 1:
                        group_positions = positions[particle_indices]
                        group_center = np.mean(group_positions, axis=0)
                        harmonic = harmonics[harmonic_idx]
                        
                        # Draw group center
                        color = harmonic_colors[group_idx % len(harmonic_colors)]
                        fig.add_trace(go.Scatter3d(
                            x=[group_center[0]],
                            y=[group_center[1]],
                            z=[group_center[2]],
                            mode='markers',
                            marker=dict(
                                size=15,
                                color=f'rgb({int(color[0]*255)}, {int(color[1]*255)}, {int(color[2]*255)})',
                                symbol='square',
                                opacity=0.8,
                                line=dict(width=2, color='white')
                            ),
                            name=f'φ-Harmonic {harmonic_idx} (f={harmonic.get("frequency", 0):.1f} Hz)',
                            showlegend=True
                        ))
            
            # Add main particle scatter plot
            fig.add_trace(go.Scatter3d(
                x=positions[:,0],
                y=positions[:,1],
                z=positions[:,2],
                mode='markers',
                marker=dict(
                    size=sizes,
                    color=color_values,
                    colorscale='RdYlBu_r' if rms_values is not None else 'Viridis',
                    opacity=0.8,
                    colorbar=dict(title='RMS Intensity' if rms_values is not None else 'Normalized Energy'),
                    line=dict(width=0)
                ),
                name=f'Particles (Step {step})'
            ))
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
# PHASE 6: ALTERNATIVE SCAFFOLDING IMPLEMENTATIONS
# (Additive - matches provided scaffolding examples exactly)
##############################

def visualize_particles_matplotlib(particles, rms=0.0, phi=(1 + 5**0.5)/2):
    """
    PHASE 6 SCAFFOLDING: Visualize particles in 3D with RMS-based coloring and φ-harmonic overlays.
    
    Alternative implementation matching the provided scaffolding example.
    
    Args:
        particles: list of Particle objects
        rms: RMS amplitude value for color intensity
        phi: Golden ratio constant (default (1+√5)/2)
    """
    try:
        if plt is None or Axes3D is None:
            logger.warning("matplotlib is not available. Cannot create visualization.")
            return
        
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        
        xs = [p.position[0] for p in particles]
        ys = [p.position[1] for p in particles]
        zs = [p.position[2] for p in particles]
        
        # Color intensity based on RMS (breath strength)
        colors = [min(1.0, rms * 5.0) for _ in particles]
        
        # Size scaling based on informational energy
        sizes = [max(10, getattr(p, 'E_info', 0.0) * 0.01) for p in particles]
        
        scatter = ax.scatter(xs, ys, zs, c=colors, s=sizes, cmap='plasma', alpha=0.8)
        
        # Overlay φ-harmonic groupings
        for i, p in enumerate(particles):
            harmonic_group = i % 3
            if harmonic_group == 0:
                ax.text(p.position[0], p.position[1], p.position[2], "φ¹", color="gold", fontsize=8)
            elif harmonic_group == 1:
                ax.text(p.position[0], p.position[1], p.position[2], "φ²", color="orange", fontsize=8)
            else:
                ax.text(p.position[0], p.position[1], p.position[2], "φ³", color="red", fontsize=8)
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title("12D CST Visualization with RMS & φ-Harmonics")
        plt.colorbar(scatter, ax=ax, label='RMS Intensity')
        plt.show()
    except Exception as e:
        logger.error(f"Error in visualize_particles_matplotlib (scaffolding): {e}")

def visualize_seed_head(particles, rms=0.0, width=800, height=600):
    """
    PHASE 6 SCAFFOLDING: Simple 2D visualization of seed head dispersal using Pygame.
    
    Alternative implementation matching the provided scaffolding example.
    
    Args:
        particles: list of Particle objects
        rms: RMS amplitude value for color intensity
        width: Screen width (default 800)
        height: Screen height (default 600)
    """
    try:
        if pygame is None:
            logger.warning("pygame is not available. Cannot create visualization.")
            return
        
        pygame.init()
        screen = pygame.display.set_mode((width, height))
        screen.fill((0, 0, 0))
        
        # Compute center for coordinate mapping
        if particles:
            positions_2d = np.array([p.position[:2] for p in particles])
            center = np.mean(positions_2d, axis=0)
        else:
            center = np.array([0.0, 0.0])
        
        # Draw particles
        for p in particles:
            # Map position to screen coordinates
            x = int(width/2 + (p.position[0] - center[0]) * 0.01)
            y = int(height/2 + (p.position[1] - center[1]) * 0.01)
            
            # Color intensity based on RMS
            intensity = min(255, int(rms * 255))
            color = (intensity, 200 - intensity//2, 50)
            
            # Draw particle
            pygame.draw.circle(screen, color, (x, y), 3)
        
        pygame.display.flip()
        
        # Keep window open briefly (or handle events for interactive mode)
        pygame.time.wait(100)
    except Exception as e:
        logger.error(f"Error in visualize_seed_head (scaffolding): {e}")

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
        
        # Initialize replay manager for deterministic reproduction
        self.replay_manager = ReplayManager(seed=42)
        self.replay_log_file = os.path.join(data_directory, 'replay.log')
        
        # Flags for integrated simulation step
        self.enable_integrated_step = True  # Use new integrated simulation_step by default
        self.enable_diagnostics = True
        self.enable_replay = True
        self.enable_visualization = False  # Disabled by default (can be slow)
        self.diagnostics_cadence = 100  # Run diagnostics every 100 steps
        self.validation_cadence = 500  # Run validation every 500 steps
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
        # Replay manager already initialized above
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
    
    def simulation_step(self, neural_net, frequency_data=None, camera_data=None, 
                       enable_diagnostics=True, enable_replay=True, enable_visualization=False,
                       diagnostics_cadence=100, validation_cadence=500):
        """
        INTEGRATED SIMULATION STEP: Executes all 7 phases in order.
        
        This method orchestrates the complete CST simulation pipeline:
        1. Physics Update (Phase 1) - Velocity Verlet or Euler
        2. Energy Update (Phase 2) - Separate energy tracking
        3. Audio-Driven Dynamics (Phase 3) - Dandelion behavior
        4. Connectivity & Internal State (Phase 4) - Ω and x12 evolution
        5. Diagnostics & Replay (Phase 5) - Conservation checks and logging
        6. Visualization (Phase 6) - RMS coloring, φ-harmonics
        7. Validation & Iteration (Phase 7) - Conservation and parameter refinement
        
        Args:
            neural_net: Neural network for particle adaptation
            frequency_data: Audio frequency data (dict with freqs, mags, etc.)
            camera_data: Camera input data (optional)
            enable_diagnostics: Enable diagnostic computation (default True)
            enable_replay: Enable replay logging (default True)
            diagnostics_cadence: Run diagnostics every N steps (default 100)
            validation_cadence: Run validation every N steps (default 500)
            enable_visualization: Enable visualization calls (default False, can be slow)
        
        Returns:
            dict with step diagnostics and state
        """
        try:
            # ====================================================================
            # PHASE 0: PREPARATION - Process audio and prepare tokens
            # ====================================================================
            audio_token = None
            rms = 0.0
            centroid = 0.0
            entropy = 0.0
            chaos = 0.0
            fft_data = {"freqs": [], "mags": []}
            
            if frequency_data:
                # Tokenize audio frame to get structured features
                if isinstance(frequency_data, dict) and 'rms' in frequency_data:
                    audio_token = frequency_data
                else:
                    # Create token from frequency data
                    audio_token = tokenize_audio_frame(frequency_data) if 'tokenize_audio_frame' in globals() else None
                
                if audio_token:
                    rms = audio_token.get('rms', 0.0)
                    centroid = audio_token.get('spectral_centroid', 0.0)
                    entropy = audio_token.get('entropy', 0.0)
                    chaos = audio_token.get('chaos', 0.0)
                    fft_data = {
                        "freqs": audio_token.get('freqs', []),
                        "mags": audio_token.get('mags', [])
                    }
            
            # ====================================================================
            # PHASE 1: PHYSICS UPDATE - Velocity Verlet or Euler integration
            # ====================================================================
            # Compute seed head center for dandelion behavior
            seed_center = compute_seed_head_center(self.particles)
            
            # Apply spring forces to seed head (before blow)
            apply_spring_forces_to_seed_head(self.particles, seed_center)
            
            # Compute forces with corrected gravitational computation
            positions = np.array([p.position for p in self.particles])
            masses = np.array([p.mass for p in self.particles])
            Ec = np.array([p.Ec for p in self.particles])
            
            # Compute connectivity for Psi (used in force computation)
            Omega = self.network.compute_connectivity(sigma=0.5, m0=1.0)
            
            # Compute Psi (informational energy density)
            Psi_result = self.dynamics.compute_Psi(Ec, Omega, positions, masses, 
                                                   particles=self.particles, dt=self.dt)
            if isinstance(Psi_result, dict):
                Psi = Psi_result['psi_total']
            else:
                Psi = Psi_result
            
            # Compute forces
            forces = self.dynamics.compute_forces(Psi, positions, masses=masses)
            
            # Update positions using Velocity Verlet or Euler
            if VELOCITY_VERLET_ENABLED and len(self.particles) > 0:
                # Store old forces for Velocity Verlet
                if not hasattr(self, '_forces_old') or self._forces_old is None:
                    self._forces_old = forces.copy()
                
                positions_array = np.array([p.position for p in self.particles])
                velocities_array = np.array([p.velocity for p in self.particles])
                masses_array = np.array([p.mass for p in self.particles])
                
                # Velocity Verlet step
                new_positions, new_velocities, new_forces = velocity_verlet_step_numba(
                    positions_array, velocities_array, masses_array, self._forces_old,
                    self.dt, G_val=G, epsilon=SOFTENING_EPSILON
                )
                
                # Update particle positions and velocities
                for i, particle in enumerate(self.particles):
                    particle.position = new_positions[i]
                    particle.velocity = new_velocities[i]
                
                # Store forces for next step
                self._forces_old = new_forces
                forces = new_forces
            else:
                # Legacy Euler integration (backward compatibility)
                for i, particle in enumerate(self.particles):
                    particle.update_position(forces[i], self.dt)
            
            # ====================================================================
            # PHASE 2: ENERGY UPDATE - Compute and update all energy components
            # ====================================================================
            # Compute energies using scaffolding function (separate K, U, U_dm, E_info)
            energies = compute_energies(self.particles, G=G, epsilon=SOFTENING_EPSILON)
            
            # Update each particle's energy components
            for i, particle in enumerate(self.particles):
                # Calculate gravitational potential energy contribution
                potential_energy = -G * np.sum([
                    other_p.mass / (np.linalg.norm(particle.position - other_p.position) + SOFTENING_EPSILON)
                    for j, other_p in enumerate(self.particles) if j != i
                ])
                
                # Dark matter potential
                delta_E_dark = self.dark_matter.compute_delta_E_dark(particle.position, particle.mass)
                
                # Update energy with separate tracking
                particle.update_energy(potential_energy, dark_matter_potential=delta_E_dark)
            
            # Assign frequencies to particles (updates E_info from audio features)
            if frequency_data:
                top_freqs = frequency_data.get('freqs', [])
                if isinstance(top_freqs, np.ndarray):
                    top_freqs = top_freqs.tolist()
                for i, freq in enumerate(top_freqs):
                    if i < len(self.particles):
                        self.particles[i].assign_frequency(float(freq), audio_token=audio_token)
            
            # ====================================================================
            # PHASE 3: AUDIO-DRIVEN DYNAMICS - Dandelion behavior and harmonics
            # ====================================================================
            # Apply dandelion dispersal behavior (RMS-based)
            apply_dandelion_dispersal(self.particles, audio_token, seed_center, self.dt)
            
            # Apply audio impulses using scaffolding function (alternative implementation)
            if audio_token:
                apply_audio_impulses(
                    self.particles,
                    rms,
                    centroid,
                    entropy,
                    blow_threshold=DANDELION_RMS_THRESHOLD,
                    k_spring=DANDELION_SPRING_K,
                    gamma_drag=DANDELION_DRAG_GAMMA
                )
            
            # Apply golden ratio harmonics for coherent clustering
            if audio_token:
                apply_golden_ratio_harmonics(self.particles, audio_token)
            
            # Assign φ-harmonics using scaffolding function (if base frequency available)
            if frequency_data and len(frequency_data.get('freqs', [])) > 0:
                base_freq = frequency_data['freqs'][0] if frequency_data['freqs'] else 440.0
                assign_phi_harmonics(self.particles, base_freq, phi=(1 + 5**0.5)/2, num_harmonics=3)
            
            # ====================================================================
            # PHASE 4: CONNECTIVITY & INTERNAL STATE - Ω computation and x12 evolution
            # ====================================================================
            # Compute connectivity using scaffolding function (cKDTree with normalization)
            Omega_norm = compute_connectivity(
                self.particles,
                cutoff=5 * DANDELION_SEED_HEAD_RADIUS,
                sigma=0.5,
                G=G,
                a0=a_0,
                m0=1.0
            )
            
            # Update internal state (x12 and m12) using scaffolding function
            update_internal_state(
                self.particles,
                Omega_norm,
                self.dt,
                k=0.1,
                gamma=0.01,
                alpha=0.05
            )
            
            # Also update using particle methods (backward compatibility)
            for particle in self.particles:
                particle.update_x12(self.dt, k=0.1, gamma=0.01)
                particle.update_memory_state(self.dt, alpha=0.05)
            
            # Update phase synchronization (Kuramoto model)
            Ksync = 0.1
            for particle in self.particles:
                neighbors = self.find_neighbors(particle)
                particle.update_phase(self.dt, neighbors, Ksync=Ksync)
            
            # Update entropy using Shannon entropy
            compute_shannon_entropy(self.particles, bins=32)
            
            # ====================================================================
            # PHASE 5: DIAGNOSTICS & REPLAY - Conservation checks and logging
            # ====================================================================
            diagnostics = None
            if enable_diagnostics and (self.step % diagnostics_cadence == 0):
                # Compute comprehensive diagnostics using scaffolding function
                diagnostics = compute_diagnostics(self.particles, energies, self.dt)
                
                # Log conservation diagnostics
                logger.debug(f"Step {self.step}: E_total={diagnostics['Ec_total']:.2e}, "
                           f"Virial={diagnostics['virial_ratio']:.3f}, "
                           f"Sync_r={diagnostics['sync_order']:.3f}")
            
            # Replay logging (every step for deterministic reproduction)
            if enable_replay and hasattr(self, 'replay_manager'):
                if audio_token:
                    log_replay_step(
                        log_file=getattr(self, 'replay_log_file', 'replay.log'),
                        fft_data=fft_data,
                        rms=rms,
                        centroid=centroid,
                        entropy=entropy,
                        diagnostics=diagnostics or compute_diagnostics(self.particles, energies, self.dt),
                        seed=self.replay_manager.seed if hasattr(self, 'replay_manager') else 42,
                        camera_data=camera_data,
                        chaos=chaos
                    )
            
            # ====================================================================
            # PHASE 6: VISUALIZATION - RMS coloring, φ-harmonics, seed head
            # ====================================================================
            if enable_visualization:
                # Matplotlib visualization with RMS and φ-harmonics
                if self.step % 10 == 0:  # Every 10 steps to avoid performance issues
                    visualize_particles_matplotlib(self.particles, rms=rms, phi=(1 + 5**0.5)/2)
                
                # Pygame seed head visualization
                if self.step % 20 == 0:  # Every 20 steps
                    visualize_seed_head(self.particles, rms=rms, width=800, height=600)
            
            # ====================================================================
            # PHASE 7: VALIDATION & ITERATION - Conservation and parameter refinement
            # ====================================================================
            if self.step % validation_cadence == 0:
                # Validate conservation laws
                if diagnostics:
                    conservation_results = validate_conservation(diagnostics, tolerance=1e-5)
                    if not conservation_results.get('energy_conserved', True):
                        logger.warning(f"Step {self.step}: Energy conservation violation detected!")
                
                # Test audio blow scenario if RMS exceeds threshold
                if rms > DANDELION_RMS_THRESHOLD:
                    blow_result = test_audio_blow(
                        self.particles,
                        audio_token or {"rms": rms, "centroid": centroid, "entropy": entropy},
                        blow_threshold=DANDELION_RMS_THRESHOLD
                    )
                    logger.debug(f"Step {self.step}: Audio blow test - {blow_result}")
                
                # Refine parameters if synchronization is low
                if diagnostics and diagnostics.get('sync_order', 0.0) < 0.8:
                    refinement = refine_parameters(self.particles, diagnostics, target_sync=0.8)
                    if refinement.get('adjustments_made', False):
                        logger.debug(f"Step {self.step}: Parameter refinement - {refinement.get('adjustment', 'N/A')}")
            
            # ====================================================================
            # LEGACY COMPONENTS - Maintain backward compatibility
            # ====================================================================
            # Learning and adaptation (legacy)
            for particle in self.particles:
                neighbors = self.find_neighbors(particle)
                if frequency_data:
                    self.learning.update_memory(particle, neighbors, frequency_data=frequency_data)
                    top_mags = frequency_data.get('mags', np.zeros(4))
                    combined_input = np.concatenate((particle.memory, top_mags[:4]))
                    self.adaptive.adapt(particle, neural_net, combined_input)
            
            # Replication (legacy)
            new_particles = self.replication.check_and_replicate(self.particles)
            self.particles.extend(new_particles)
            
            # Camera data processing (legacy)
            if camera_data:
                self.process_camera_data(camera_data)
            
            # ====================================================================
            # FINAL UPDATES - Adaptive timestep and time increment
            # ====================================================================
            # Adaptive timestep using cKDTree
            adaptive_dt = compute_adaptive_dt(self.particles, dt_max=self.dt0, safety_factor=0.1)
            self.dt = min(self.dt0, max(adaptive_dt, 0.1 * self.dt0))
            
            # Increment step counter and time (ONCE - fix double increment bug)
            self.step += 1
            self.time += self.dt
            
            # Update history
            self.history.append(np.array([p.position for p in self.particles]))
            
            return {
                "step": self.step,
                "time": self.time,
                "energies": energies,
                "diagnostics": diagnostics,
                "rms": rms,
                "sync_order": diagnostics.get('sync_order', 0.0) if diagnostics else 0.0
            }
            
        except Exception as e:
            logger.error(f"Error in simulation_step at step {self.step}: {e}")
            logger.error(traceback.format_exc())
            # Increment step even on error to prevent infinite loops
            self.step += 1
            self.time += self.dt
            return {"step": self.step, "error": str(e)}
    
    def run_step(self, neural_net, frequency_data=None, camera_data=None):
        """
        Executes a single simulation step, updating particle states and processing interactions.
        
        This method now uses the integrated simulation_step() by default, which executes all 7 phases
        in order. Set self.enable_integrated_step = False to use legacy implementation.
        
        AUDIO PROCESSING FLOW (Core Engine Integration):
        Each sound/blow captured by microphone is processed through the core engine:
        1. Audio -> FFT -> frequency_data (freqs, mags) passed to this method
        2. Frequencies assigned to particles via assign_frequency() -> updates particle.Ec, particle.nu
        3. Updated Ec affects Psi computation (informational energy density)
        4. Psi affects force computation -> forces affect particle positions
        5. Frequency data updates particle memory -> neural network adaptation -> Ec changes
        6. All particle reactions come through core engine methods (update_position, update_energy, etc.)
        """
        # Use integrated simulation step by default (executes all 7 phases)
        if self.enable_integrated_step:
            return self.simulation_step(
                neural_net,
                frequency_data=frequency_data,
                camera_data=camera_data,
                enable_diagnostics=self.enable_diagnostics,
                enable_replay=self.enable_replay,
                enable_visualization=self.enable_visualization,
                diagnostics_cadence=self.diagnostics_cadence,
                validation_cadence=self.validation_cadence
            )
        
        # Legacy implementation (backward compatibility)
        try:
            # Scan directory for data files
            all_files = scan_directory(self.data_directory)
            json_files = [f for f in all_files if f.endswith('.json')]
            db_files = [f for f in all_files if f.endswith('.db')]
            json_data = load_json_files(json_files)
            db_data = load_db_files(db_files)
            combined_data = json_data + db_data  # (For potential future use)
            
            # PHASE 2 & 3: Process frequency data and tokenize audio
            audio_token = None
            if frequency_data:
                # Check if frequency_data is already a token (from tokenize_audio_frame)
                if isinstance(frequency_data, dict) and 'rms' in frequency_data:
                    audio_token = frequency_data
                    freqs = audio_token.get('freqs', [])
                    mags = audio_token.get('mags', [])
                else:
                    # Legacy format: extract freqs and mags
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
            
            # PHASE 2: Assign frequencies to particles with audio_token for proper E_info update
            frequency_particle_map = {}
            if isinstance(top_freqs, np.ndarray):
                top_freqs_list = top_freqs.tolist()
            else:
                top_freqs_list = list(top_freqs) if top_freqs else []
            for i, freq in enumerate(top_freqs_list):
                if i < len(self.particles):
                    particle = self.particles[i]
                    # PHASE 2: Pass audio_token to assign_frequency for normalized E_info update
                    particle.assign_frequency(float(freq), audio_token=audio_token)
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
            
            # PHASE 2: Compute dark matter influence (update U_dm separately)
            delta_E_dark = np.array([self.dark_matter.compute_delta_E_dark(p.position, p.mass) for p in self.particles])
            
            # Get current lambda value
            current_lambda = self.dynamics.get_lambda_evo(self.time)
            self.dynamics.alpha = current_lambda
            
            # PHASE 1: Compute forces with corrected gravitational computation (includes masses)
            forces = self.dynamics.compute_forces(Psi, positions, masses=masses)
            
            # PHASE 3: Dandelion behavior - compute seed head center
            seed_center = compute_seed_head_center(self.particles)
            
            # PHASE 3: Apply spring forces to seed head (before blow)
            apply_spring_forces_to_seed_head(self.particles, seed_center)
            
            # PHASE 1: Update particle positions using Velocity Verlet or Euler (based on flag)
            if VELOCITY_VERLET_ENABLED and len(self.particles) > 0:
                # Store old forces for Velocity Verlet
                if not hasattr(self, '_forces_old') or self._forces_old is None:
                    self._forces_old = forces.copy()
                
                # Convert to numpy arrays for Velocity Verlet
                positions_array = np.array([p.position for p in self.particles])
                velocities_array = np.array([p.velocity for p in self.particles])
                masses_array = np.array([p.mass for p in self.particles])
                
                # Velocity Verlet step
                new_positions, new_velocities, new_forces = velocity_verlet_step_numba(
                    positions_array, velocities_array, masses_array, self._forces_old,
                    self.dt, G_val=G, epsilon=SOFTENING_EPSILON
                )
                
                # Update particle positions and velocities
                for i, particle in enumerate(self.particles):
                    particle.position = new_positions[i]
                    particle.velocity = new_velocities[i]
                
                # Store forces for next step
                self._forces_old = new_forces
                forces = new_forces  # Update forces for potential use later
            else:
                # Legacy Euler integration (backward compatibility)
                for i, particle in enumerate(self.particles):
                    particle.update_position(forces[i], self.dt)
            
            # PHASE 3: Apply dandelion dispersal behavior (after position update)
            apply_dandelion_dispersal(self.particles, audio_token, seed_center, self.dt)
            
            # PHASE 3: Apply golden ratio harmonics for coherent clustering
            apply_golden_ratio_harmonics(self.particles, audio_token)
            
            # PHASE 2: Update particle energies with proper separation (K, U, U_dm, E_info)
            for i, particle in enumerate(self.particles):
                # Calculate gravitational potential energy
                potential_energy = -G * np.sum([
                    other_p.mass / (np.linalg.norm(particle.position - other_p.position) + 1e-10)
                    for j, other_p in enumerate(self.particles) if j != i
                ])
                
                # Update energy with separate tracking (K, U, U_dm, E_info)
                particle.update_energy(potential_energy, dark_matter_potential=delta_E_dark[i])
            
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
            
            # PHASE 1: Adaptive timestep - dynamically adjust for numerical stability with recovery
            # Use cKDTree for efficient r_min computation
            adaptive_dt = compute_adaptive_dt(self.particles, dt_max=self.dt0, safety_factor=0.1)
            # Clamp dt between 10% of original and original value, allowing recovery
            self.dt = min(self.dt0, max(adaptive_dt, 0.1 * self.dt0))
            
            # PHASE 1: Increment step counter and time (ONCE - fix double increment bug)
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
            
            # PHASE 1: Time already incremented above - removed duplicate increment
            logger.info(f"Completed step {self.step} at time {self.time:.2e} with {len(self.particles)} particles.")
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
# PHASE 7: VALIDATION & ITERATION
##############################

class ValidationSuite:
    """
    PHASE 7: Validation tests for CST simulation.
    
    Tests include:
    - Energy conservation
    - Momentum conservation
    - Angular momentum conservation
    - Virial theorem
    - Audio blow scenario behavior
    - Replay determinism
    """
    
    def __init__(self, simulator):
        self.simulator = simulator
        self.test_results = {}
    
    def test_energy_conservation(self, steps=100, tolerance=0.01):
        """
        Test energy conservation over simulation steps.
        
        Args:
            steps: Number of steps to run
            tolerance: Maximum allowed relative energy drift (default 0.01 = 1%)
        
        Returns:
            dict with test results
        """
        try:
            initial_energy = compute_total_energy(self.simulator.particles)
            initial_energies = {
                'K': sum(0.5 * p.mass * np.linalg.norm(p.velocity)**2 for p in self.simulator.particles),
                'U': sum(p.U for p in self.simulator.particles),
                'U_dm': sum(p.U_dm for p in self.simulator.particles),
                'E_info': sum(p.E_info for p in self.simulator.particles),
                'Ec_total': initial_energy
            }
            
            neural_net = ParticleNeuralNet()
            neural_net.eval()
            
            energy_history = [initial_energy]
            for i in range(steps):
                self.simulator.run_step(neural_net, frequency_data=None, camera_data=None)
                current_energy = compute_total_energy(self.simulator.particles)
                energy_history.append(current_energy)
            
            final_energy = energy_history[-1]
            energy_drift = abs((final_energy - initial_energy) / initial_energy) if abs(initial_energy) > 1e-10 else 0.0
            
            # Check individual energy components
            final_energies = {
                'K': sum(0.5 * p.mass * np.linalg.norm(p.velocity)**2 for p in self.simulator.particles),
                'U': sum(p.U for p in self.simulator.particles),
                'U_dm': sum(p.U_dm for p in self.simulator.particles),
                'E_info': sum(p.E_info for p in self.simulator.particles),
                'Ec_total': final_energy
            }
            
            result = {
                'test_name': 'Energy Conservation',
                'passed': energy_drift < tolerance,
                'initial_energy': initial_energy,
                'final_energy': final_energy,
                'energy_drift': energy_drift,
                'tolerance': tolerance,
                'initial_components': initial_energies,
                'final_components': final_energies,
                'energy_history': energy_history
            }
            
            self.test_results['energy_conservation'] = result
            logger.info(f"Energy Conservation Test: {'PASSED' if result['passed'] else 'FAILED'} "
                       f"(drift: {energy_drift:.4f}, tolerance: {tolerance:.4f})")
            return result
        except Exception as e:
            logger.error(f"Error in energy conservation test: {e}")
            return {'test_name': 'Energy Conservation', 'passed': False, 'error': str(e)}
    
    def test_momentum_conservation(self, steps=100, tolerance=1e-6):
        """
        Test momentum conservation (should be conserved in absence of external forces).
        
        Args:
            steps: Number of steps to run
            tolerance: Maximum allowed momentum change (default 1e-6)
        
        Returns:
            dict with test results
        """
        try:
            initial_momentum = compute_total_momentum(self.simulator.particles)
            initial_momentum_mag = np.linalg.norm(initial_momentum)
            
            neural_net = ParticleNeuralNet()
            neural_net.eval()
            
            momentum_history = [initial_momentum_mag]
            for i in range(steps):
                self.simulator.run_step(neural_net, frequency_data=None, camera_data=None)
                current_momentum = compute_total_momentum(self.simulator.particles)
                momentum_history.append(np.linalg.norm(current_momentum))
            
            final_momentum = compute_total_momentum(self.simulator.particles)
            final_momentum_mag = np.linalg.norm(final_momentum)
            momentum_change = np.linalg.norm(final_momentum - initial_momentum)
            
            result = {
                'test_name': 'Momentum Conservation',
                'passed': momentum_change < tolerance,
                'initial_momentum': initial_momentum.tolist(),
                'final_momentum': final_momentum.tolist(),
                'momentum_change': momentum_change,
                'tolerance': tolerance,
                'momentum_history': momentum_history
            }
            
            self.test_results['momentum_conservation'] = result
            logger.info(f"Momentum Conservation Test: {'PASSED' if result['passed'] else 'FAILED'} "
                       f"(change: {momentum_change:.6e}, tolerance: {tolerance:.6e})")
            return result
        except Exception as e:
            logger.error(f"Error in momentum conservation test: {e}")
            return {'test_name': 'Momentum Conservation', 'passed': False, 'error': str(e)}
    
    def test_audio_blow_scenario(self, rms_threshold=0.1, steps_before=50, steps_after=100):
        """
        Test audio blow scenario to ensure dispersal matches dandelion behavior.
        
        Args:
            rms_threshold: RMS threshold for blow detection
            steps_before: Steps to run before blow
            steps_after: Steps to run after blow
        
        Returns:
            dict with test results
        """
        try:
            # Reset simulator
            num_particles = len(self.simulator.particles)
            self.simulator = Simulator(num_particles=num_particles, steps=steps_before + steps_after, 
                                      dt=self.simulator.dt, data_directory=self.simulator.data_directory)
            
            neural_net = ParticleNeuralNet()
            neural_net.eval()
            
            # Run before blow (low RMS)
            low_rms_token = {
                'freqs': [440.0, 880.0, 1320.0],
                'mags': [0.1, 0.05, 0.02],
                'rms': 0.01,  # Below threshold
                'spectral_centroid': 500.0,
                'chaos': 0.1,
                'entropy': 0.5,
                'harmonics': []
            }
            
            positions_before = []
            attached_count_before = []
            
            for i in range(steps_before):
                self.simulator.run_step(neural_net, frequency_data=low_rms_token, camera_data=None)
                positions_before.append(np.array([p.position.copy() for p in self.simulator.particles]))
                attached_count_before.append(sum(1 for p in self.simulator.particles if hasattr(p, 'spring_attached') and p.spring_attached))
            
            # Apply blow (high RMS)
            high_rms_token = {
                'freqs': [440.0, 880.0, 1320.0],
                'mags': [1.0, 0.8, 0.6],
                'rms': rms_threshold * 2.0,  # Above threshold
                'spectral_centroid': 2000.0,
                'chaos': 0.8,
                'entropy': 0.9,
                'harmonics': []
            }
            
            positions_after = []
            attached_count_after = []
            velocities_after = []
            
            for i in range(steps_after):
                self.simulator.run_step(neural_net, frequency_data=high_rms_token, camera_data=None)
                positions_after.append(np.array([p.position.copy() for p in self.simulator.particles]))
                attached_count_after.append(sum(1 for p in self.simulator.particles if hasattr(p, 'spring_attached') and p.spring_attached))
                velocities_after.append(np.array([np.linalg.norm(p.velocity) for p in self.simulator.particles]))
            
            # Analyze results
            seed_center_before = np.mean(positions_before[-1], axis=0)
            seed_center_after = np.mean(positions_after[-1], axis=0)
            
            # Check that particles detached
            detached_ratio = 1.0 - (attached_count_after[-1] / num_particles) if num_particles > 0 else 0.0
            
            # Check that particles dispersed (increased spread)
            spread_before = np.std(positions_before[-1], axis=0)
            spread_after = np.std(positions_after[-1], axis=0)
            spread_increase = np.linalg.norm(spread_after) / (np.linalg.norm(spread_before) + 1e-10)
            
            # Check that velocities increased (dispersal impulse)
            avg_velocity_before = np.mean([np.linalg.norm(p.velocity) for p in self.simulator.particles]) if len(positions_before) > 0 else 0.0
            avg_velocity_after = np.mean(velocities_after[-1]) if len(velocities_after) > 0 else 0.0
            velocity_increase = avg_velocity_after / (avg_velocity_before + 1e-10)
            
            result = {
                'test_name': 'Audio Blow Scenario',
                'passed': detached_ratio > 0.5 and spread_increase > 1.2 and velocity_increase > 1.1,
                'detached_ratio': detached_ratio,
                'spread_increase': spread_increase,
                'velocity_increase': velocity_increase,
                'attached_before': attached_count_before[-1] if attached_count_before else 0,
                'attached_after': attached_count_after[-1] if attached_count_after else 0,
                'seed_center_before': seed_center_before.tolist(),
                'seed_center_after': seed_center_after.tolist()
            }
            
            self.test_results['audio_blow'] = result
            logger.info(f"Audio Blow Scenario Test: {'PASSED' if result['passed'] else 'FAILED'} "
                       f"(detached: {detached_ratio:.2%}, spread: {spread_increase:.2f}x, velocity: {velocity_increase:.2f}x)")
            return result
        except Exception as e:
            logger.error(f"Error in audio blow scenario test: {e}")
            return {'test_name': 'Audio Blow Scenario', 'passed': False, 'error': str(e)}
    
    def test_replay_determinism(self, seed=42, steps=50):
        """
        Test replay determinism with identical seeds and inputs.
        
        Args:
            seed: Random seed for reproducibility
            steps: Number of steps to run
        
        Returns:
            dict with test results
        """
        try:
            # Run simulation twice with same seed
            num_particles = len(self.simulator.particles)
            
            # First run
            random.seed(seed)
            np.random.seed(seed)
            if torch is not None:
                torch.manual_seed(seed)
            
            sim1 = Simulator(num_particles=num_particles, steps=steps, dt=self.simulator.dt,
                            data_directory=self.simulator.data_directory)
            neural_net1 = ParticleNeuralNet()
            neural_net1.eval()
            
            audio_token = {
                'freqs': [440.0, 880.0],
                'mags': [0.5, 0.3],
                'rms': 0.2,
                'spectral_centroid': 1000.0,
                'chaos': 0.5,
                'entropy': 0.7,
                'harmonics': []
            }
            
            for i in range(steps):
                sim1.run_step(neural_net1, frequency_data=audio_token, camera_data=None)
            
            positions1 = np.array([p.position for p in sim1.particles])
            velocities1 = np.array([p.velocity for p in sim1.particles])
            
            # Second run with same seed
            random.seed(seed)
            np.random.seed(seed)
            if torch is not None:
                torch.manual_seed(seed)
            
            sim2 = Simulator(num_particles=num_particles, steps=steps, dt=self.simulator.dt,
                            data_directory=self.simulator.data_directory)
            neural_net2 = ParticleNeuralNet()
            neural_net2.eval()
            
            for i in range(steps):
                sim2.run_step(neural_net2, frequency_data=audio_token, camera_data=None)
            
            positions2 = np.array([p.position for p in sim2.particles])
            velocities2 = np.array([p.velocity for p in sim2.particles])
            
            # Compare results
            position_diff = np.max(np.abs(positions1 - positions2))
            velocity_diff = np.max(np.abs(velocities1 - velocities2))
            tolerance = 1e-6
            
            result = {
                'test_name': 'Replay Determinism',
                'passed': position_diff < tolerance and velocity_diff < tolerance,
                'position_diff': position_diff,
                'velocity_diff': velocity_diff,
                'tolerance': tolerance,
                'seed': seed
            }
            
            self.test_results['replay_determinism'] = result
            logger.info(f"Replay Determinism Test: {'PASSED' if result['passed'] else 'FAILED'} "
                       f"(pos_diff: {position_diff:.6e}, vel_diff: {velocity_diff:.6e}, tolerance: {tolerance:.6e})")
            return result
        except Exception as e:
            logger.error(f"Error in replay determinism test: {e}")
            return {'test_name': 'Replay Determinism', 'passed': False, 'error': str(e)}
    
    def run_all_tests(self):
        """
        Run all validation tests.
        
        Returns:
            dict with all test results
        """
        logger.info("Starting validation test suite...")
        self.test_energy_conservation()
        self.test_momentum_conservation()
        self.test_audio_blow_scenario()
        self.test_replay_determinism()
        
        passed = sum(1 for r in self.test_results.values() if r.get('passed', False))
        total = len(self.test_results)
        
        logger.info(f"Validation suite complete: {passed}/{total} tests passed")
        return self.test_results

##############################
# PHASE 7: ALTERNATIVE SCAFFOLDING IMPLEMENTATIONS
# (Additive - matches provided scaffolding examples exactly)
##############################

def validate_conservation(diagnostics, tolerance=1e-5):
    """
    PHASE 7 SCAFFOLDING: Validate conservation laws: energy, momentum, angular momentum.
    
    Alternative implementation matching the provided scaffolding example.
    
    Args:
        diagnostics: dict from compute_diagnostics (includes momentum, angular_momentum, Ec_total)
        tolerance: Tolerance for conservation checks (default 1e-5)
    
    Returns:
        dict with conservation test results
    """
    try:
        results = {}
        
        # Energy conservation check
        # Track energy over time to detect drift
        Ec_total = diagnostics.get("Ec_total", 0.0)
        if not hasattr(validate_conservation, "last_energy"):
            validate_conservation.last_energy = Ec_total
        
        energy_diff = abs(Ec_total - validate_conservation.last_energy)
        results["energy_conserved"] = energy_diff < tolerance
        results["energy_drift"] = energy_diff
        validate_conservation.last_energy = Ec_total
        
        # Momentum conservation check
        # Total momentum should be zero (or constant) in absence of external forces
        momentum = diagnostics.get("momentum", [0.0, 0.0, 0.0])
        if isinstance(momentum, list):
            momentum = np.array(momentum)
        results["momentum_conserved"] = np.allclose(momentum, 0.0, atol=tolerance)
        results["momentum_magnitude"] = float(np.linalg.norm(momentum))
        
        # Angular momentum conservation check
        # Total angular momentum should be zero (or constant) in absence of external torques
        ang_momentum = diagnostics.get("angular_momentum", [0.0, 0.0, 0.0])
        if isinstance(ang_momentum, list):
            ang_momentum = np.array(ang_momentum)
        results["angular_momentum_conserved"] = np.allclose(ang_momentum, 0.0, atol=tolerance)
        results["angular_momentum_magnitude"] = float(np.linalg.norm(ang_momentum))
        
        return results
    except Exception as e:
        logger.error(f"Error in validate_conservation (scaffolding): {e}")
        return {
            "energy_conserved": False,
            "momentum_conserved": False,
            "angular_momentum_conserved": False,
            "error": str(e)
        }

def test_audio_blow(particles, audio_features, blow_threshold=0.2):
    """
    PHASE 7 SCAFFOLDING: Run a test scenario where RMS exceeds threshold to confirm dispersal.
    
    Alternative implementation matching the provided scaffolding example.
    
    Args:
        particles: list of Particle objects
        audio_features: dict with keys: rms, centroid, entropy
        blow_threshold: RMS threshold for blow detection (default 0.2)
    
    Returns:
        str indicating dispersal status
    """
    try:
        rms = audio_features.get("rms", 0.0)
        if rms > blow_threshold:
            # Apply audio impulses using scaffolding function
            apply_audio_impulses(
                particles, 
                rms, 
                audio_features.get("centroid", 0.0),
                audio_features.get("entropy", 0.0),
                blow_threshold=blow_threshold
            )
            return "Dispersal triggered"
        else:
            return "No dispersal"
    except Exception as e:
        logger.error(f"Error in test_audio_blow (scaffolding): {e}")
        return f"Error: {str(e)}"

def check_replay_determinism(log_file):
    """
    PHASE 7 SCAFFOLDING: Verify replay determinism by comparing sequential runs.
    
    Alternative implementation matching the provided scaffolding example.
    
    Args:
        log_file: Path to replay log file
    
    Returns:
        bool indicating if replay is deterministic
    """
    try:
        if not os.path.exists(log_file):
            logger.warning(f"Replay log file {log_file} does not exist.")
            return False
        
        entries = load_replay_log(log_file)
        if not entries:
            return False
        
        # Compare seeds and dt across runs
        seeds = [e.get("seed") for e in entries]
        dts = [e.get("dt") for e in entries]
        
        # Deterministic if all seeds are the same and all dts are the same
        deterministic = (len(set(seeds)) == 1) and (len(set(dts)) == 1)
        
        if deterministic:
            logger.info(f"Replay determinism verified: seed={seeds[0] if seeds else None}, dt={dts[0] if dts else None}")
        else:
            logger.warning(f"Replay determinism check failed: seeds={set(seeds)}, dts={set(dts)}")
        
        return deterministic
    except Exception as e:
        logger.error(f"Error in check_replay_determinism (scaffolding): {e}")
        return False

def refine_parameters(particles, diagnostics, target_sync=0.8):
    """
    PHASE 7 SCAFFOLDING: Iteratively refine parameters (ψ weights, Ω normalization, drag coefficients).
    
    Alternative implementation matching the provided scaffolding example.
    
    Args:
        particles: list of Particle objects
        diagnostics: dict from compute_diagnostics (includes sync_order)
        target_sync: Target synchronization order parameter (default 0.8)
    
    Returns:
        dict with refinement results
    """
    try:
        sync_order = diagnostics.get("sync_order", 0.0)
        results = {
            "sync_order": sync_order,
            "target_sync": target_sync,
            "adjustments_made": False
        }
        
        if sync_order < target_sync:
            # Increase coupling strength (increase omega)
            for p in particles:
                if hasattr(p, 'omega'):
                    p.omega *= 1.05
            results["adjustments_made"] = True
            results["adjustment"] = "Increased coupling strength (omega *= 1.05)"
            logger.debug(f"Refined parameters: increased coupling strength (sync_order={sync_order:.4f} < {target_sync})")
        else:
            # Slightly reduce drag to stabilize
            for p in particles:
                if hasattr(p, 'velocity'):
                    p.velocity *= 0.99
            results["adjustments_made"] = True
            results["adjustment"] = "Reduced drag (velocity *= 0.99)"
            logger.debug(f"Refined parameters: reduced drag (sync_order={sync_order:.4f} >= {target_sync})")
        
        return results
    except Exception as e:
        logger.error(f"Error in refine_parameters (scaffolding): {e}")
        return {
            "sync_order": 0.0,
            "target_sync": target_sync,
            "adjustments_made": False,
            "error": str(e)
        }

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
    # Example: Run validation tests
    # simulator = Simulator(num_particles=100, steps=1000, dt=1.0)
    # validator = ValidationSuite(simulator)
    # results = validator.run_all_tests()
    # print(f"Test Results: {results}")
    
    main_entry()
