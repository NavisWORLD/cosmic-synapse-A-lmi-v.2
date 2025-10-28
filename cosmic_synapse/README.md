# Cosmic Synapse Simulation

Unity-based particle simulation demonstrating vibrational information dynamics through audio-driven stochastic resonance.

## Overview

The Cosmic Synapse simulation visualizes how vibrational information principles manifest in a physical system. It implements:

- **Real-Time Physics**: Particle simulation with multiple force fields
- **Audio-Driven Dynamics**: Live microphone input modulates simulation noise
- **Stochastic Resonance**: PSD-normalized audio affects particle behavior
- **Visualization**: 2D/3D rendering of vibrational information flow

## Physics Implementation

### Forces

1. **Conservative Bowl Potential** (φ):
   - Radial force toward center
   - Creates stable central basin

2. **Gravitational Potential** (U_grav):
   - Attraction to spawned masses (stars, black holes)
   - Softened to prevent singularities

3. **Connectivity Potential** (U_conn):
   - Particle-to-particle interactions
   - Dependent on local density

4. **Swirl Force** (F_swirl):
   - Perpendicular force creating rotational motion
   - Modulated by Ω parameter

5. **Damping** (F_damp):
   - Velocity-dependent drag
   - Ensures energy dissipation

### Noise Modulation

- **Microphone OFF**: Constant noise intensity σ = σ_λ·λ
- **Microphone ON**: Live PSD modulates noise σ = σ_psd·PSD_norm

## Requirements

- Unity 2022.3 LTS
- C# programming
- Microphone input (optional)

## Installation

1. Open Unity 2022.3 LTS
2. Create new 3D project
3. Import scripts from `Scripts/` directory
4. Set up scene with ParticleSystem

## Usage

1. Start simulation
2. Adjust parameters (Ω, λ, particle count)
3. Enable microphone for live audio modulation
4. Observe particle clustering and dynamics

