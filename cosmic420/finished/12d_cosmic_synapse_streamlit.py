# -*- coding: utf-8 -*-
"""
12D COSMIC SYNAPSE THEORY - STREAMLIT UI
Interactive dashboard with audio input, real-time visualization, and token export
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import time
import queue
import threading
import json
from datetime import datetime
import io

# Import the engine
import sys
import os

# Import with proper module name handling
import importlib.util
engine_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "12d_cosmic_synapse_engine.py")
spec = importlib.util.spec_from_file_location("cosmic_engine", engine_path)
cosmic_engine = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cosmic_engine)

Simulator = cosmic_engine.Simulator
Particle = cosmic_engine.Particle
AudioFrame = cosmic_engine.AudioFrame
TokenStream = cosmic_engine.TokenStream
Recorder = cosmic_engine.Recorder
SimulationMode = cosmic_engine.SimulationMode
PhysicsConfig = cosmic_engine.PhysicsConfig
AdaptiveConfig = cosmic_engine.AdaptiveConfig
SyncConfig = cosmic_engine.SyncConfig
TimestepConfig = cosmic_engine.TimestepConfig
DarkMatterParams = cosmic_engine.DarkMatterParams
C = cosmic_engine.C
PHI = cosmic_engine.PHI
H = cosmic_engine.H
KB = cosmic_engine.KB
G = cosmic_engine.G

# Try to import pyaudio, fallback to mock if not available
PYAUDIO_AVAILABLE = False
FORMAT = None
try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
    FORMAT = pyaudio.paInt16
except ImportError:
    PYAUDIO_AVAILABLE = False
    FORMAT = None

# Audio configuration
SAMPLE_RATE = 44100
CHUNK_SIZE = 4096
FFT_SIZE = 2048
CHANNELS = 1


def fft_analysis(data_array, sample_rate=SAMPLE_RATE):
    """Perform FFT analysis on audio data"""
    # Convert to numpy array
    if isinstance(data_array, list):
        data_array = np.array(data_array, dtype=np.float32)
    
    # Ensure it's 1D
    if data_array.ndim > 1:
        data_array = data_array.flatten()
    
    # Normalize (handle zero case)
    max_val = np.max(np.abs(data_array))
    if max_val > 0:
        data_array = data_array / max_val
    
    # Perform FFT
    fft = np.fft.rfft(data_array, n=FFT_SIZE)
    magnitude = np.abs(fft)
    frequencies = np.fft.rfftfreq(FFT_SIZE, 1/sample_rate)
    
    # Get top frequencies
    top_indices = np.argsort(magnitude)[-10:][::-1]
    frequency_data = [
        {"frequency": float(frequencies[i]), "magnitude": float(magnitude[i] / np.max(magnitude))}
        for i in top_indices if magnitude[i] > 0.05
    ]
    
    # Calculate RMS
    rms = np.sqrt(np.mean(data_array ** 2))
    
    # Calculate spectral centroid
    if len(frequency_data) > 0:
        weighted_sum = sum(f["frequency"] * f["magnitude"] for f in frequency_data)
        magnitude_sum = sum(f["magnitude"] for f in frequency_data)
        spectral_centroid = weighted_sum / magnitude_sum if magnitude_sum > 0 else 0
    else:
        spectral_centroid = 0.0
    
    return frequency_data, rms, spectral_centroid


def generate_phi_harmonics(fundamental, count=8):
    """Generate φ-harmonic series"""
    harmonics = []
    for i in range(count):
        freq = fundamental * (PHI ** (i / 2))
        # Octave folding
        while freq > fundamental * 4:
            freq /= 2
        while freq < fundamental / 2:
            freq *= 2
        harmonics.append(freq)
    return sorted(harmonics)


def audio_capture_thread(simulator, audio_queue, stop_event):
    """Audio capture thread"""
    if not PYAUDIO_AVAILABLE:
        # Simulate audio with sine waves - generate chunks
        t = 0
        while not stop_event.is_set():
            # Generate synthetic audio chunk
            chunk = np.array([
                np.sin(2 * np.pi * 440 * (t + i / SAMPLE_RATE)) * 0.5 + 
                np.sin(2 * np.pi * 880 * (t + i / SAMPLE_RATE)) * 0.3
                for i in range(CHUNK_SIZE)
            ], dtype=np.float32)
            audio_queue.put(chunk)
            t += CHUNK_SIZE / SAMPLE_RATE
            time.sleep(CHUNK_SIZE / SAMPLE_RATE)  # Sleep for chunk duration
        return
    
    try:
        p = pyaudio.PyAudio()
        stream = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=CHUNK_SIZE
        )
        
        while not stop_event.is_set():
            data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
            audio_data = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            audio_queue.put(audio_data)
        
        stream.stop_stream()
        stream.close()
        p.terminate()
    except Exception as e:
        st.error(f"Audio capture error: {e}")


def process_audio_thread(simulator, audio_queue, processed_queue, stop_event):
    """Process audio frames and generate tokens"""
    while not stop_event.is_set():
        try:
            # Get audio data (non-blocking)
            audio_data = audio_queue.get(timeout=0.1)
            
            # Perform FFT analysis
            frequency_data, rms, spectral_centroid = fft_analysis(audio_data)
            
            if len(frequency_data) > 0:
                # Generate harmonics
                fundamental = frequency_data[0]["frequency"]
                harmonics = generate_phi_harmonics(fundamental, 8)
                
                # Create audio frame
                frame = AudioFrame(
                    timestamp=time.time(),
                    rmsEnergy=float(rms),
                    frequencyData=frequency_data,
                    spectralCentroid=float(spectral_centroid),
                    harmonics=harmonics,
                    dataArray=audio_data
                )
                
                # Add to processed queue
                processed_queue.put(frame)
        except queue.Empty:
            continue
        except Exception as e:
            st.error(f"Audio processing error: {e}")


def initialize_session_state():
    """Initialize Streamlit session state"""
    if 'simulator' not in st.session_state:
        st.session_state.simulator = Simulator()
        st.session_state.simulator.set_seed(12345)
    
    if 'audio_running' not in st.session_state:
        st.session_state.audio_running = False
    
    if 'audio_thread' not in st.session_state:
        st.session_state.audio_thread = None
    
    if 'process_thread' not in st.session_state:
        st.session_state.process_thread = None
    
    if 'stop_event' not in st.session_state:
        st.session_state.stop_event = threading.Event()
    
    if 'last_update' not in st.session_state:
        st.session_state.last_update = time.time()
    
    if 'history' not in st.session_state:
        st.session_state.history = {
            'time': [],
            'psi_total': [],
            'sync_r': [],
            'energy': [],
            'token_rate': [],
            'particle_count': []
        }


def update_simulator_from_ui(simulator, physics_config, adapt_config, sync_config, timestep_config, dm_params, audio_sensitivity):
    """Update simulator configuration from UI"""
    simulator.physics.blendLorenz = physics_config['blend_lorenz']
    simulator.physics.gravEnabled = physics_config['grav_enabled']
    simulator.physics.dmEnabled = physics_config['dm_enabled']
    simulator.physics.epsilon = physics_config['epsilon']
    simulator.physics.rCutoff = physics_config['rcutoff']
    
    simulator.adapt.k = adapt_config['k']
    simulator.adapt.gamma = adapt_config['gamma']
    simulator.adapt.alpha = adapt_config['alpha']
    simulator.adapt.sigmaSimilarity = adapt_config['sigma_sim']
    
    simulator.sync.Ksync = sync_config['ksync']
    
    simulator.timestep.dtMax = timestep_config['dt_max']
    simulator.timestep.adaptive = timestep_config['adaptive']
    
    simulator.dm_params.rho0 = dm_params['rho0']
    simulator.dm_params.rs = dm_params['rs']
    
    # CST v2.0 additive: Audio sensitivity for real-time modulation
    simulator.audio_sensitivity = audio_sensitivity


def create_3d_scatter(particles):
    """Create 3D scatter plot of particles"""
    if len(particles) == 0:
        return go.Figure()
    
    x = [p.x for p in particles]
    y = [p.y for p in particles]
    z = [p.z for p in particles]
    
    # CST v2.0 additive: Color by frequency (sound-to-color mapping) or energy
    colors = []
    sizes = []
    for p in particles:
        if p.frequency > 0:
            # Map frequency to hue (0-360) for color mapping
            hue = (p.frequency / 20000.0) * 360.0
            colors.append(hue)
        else:
            colors.append(p.Ec)
        # Size based on mass/energy
        sizes.append(5 + p.mass * 2)
    
    fig = go.Figure(data=go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(
            size=sizes,
            color=colors,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Frequency/Energy"),
            line=dict(width=0.5, color='rgba(0,0,0,0.3)')
        ),
        text=[f"Particle {i}<br>Freq: {p.frequency:.1f}Hz<br>Ec: {p.Ec:.2e}<br>Ω: {p.omega:.3f}" 
              for i, p in enumerate(particles)],
        hovertemplate='%{text}<extra></extra>'
    ))
    
    fig.update_layout(
        title="3D Particle Visualization - Audio-Reactive",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode='cube',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
        ),
        height=600
    )
    
    return fig


def main():
    st.set_page_config(
        page_title="12D Cosmic Synapse Engine",
        page_icon="🌌",
        layout="wide"
    )
    
    st.title("🌌 12D Cosmic Synapse Theory - Interactive Engine")
    st.markdown("**Continuous Token Generation • Real-time Visualization • Deterministic Replay**")
    
    # Show warning if pyaudio not available
    if not PYAUDIO_AVAILABLE:
        st.warning("⚠️ PyAudio not installed. Audio input will be simulated. Install with: `pip install pyaudio`")
    
    # Initialize session state
    initialize_session_state()
    
    simulator = st.session_state.simulator
    
    # Sidebar controls
    with st.sidebar:
        st.header("🎛️ Controls")
        
        # Audio controls
        st.subheader("🎤 Audio")
        audio_sensitivity = st.slider("Audio Sensitivity", 0.1, 5.0, 1.0, 0.1, key="audio_sensitivity")
        
        if st.button("🎤 Start Audio" if not st.session_state.audio_running else "⏹️ Stop Audio"):
            if not st.session_state.audio_running:
                # Start audio
                st.session_state.audio_running = True
                st.session_state.stop_event.clear()
                st.session_state.audio_thread = threading.Thread(
                    target=audio_capture_thread,
                    args=(simulator, simulator.audio_queue, st.session_state.stop_event),
                    daemon=True
                )
                st.session_state.process_thread = threading.Thread(
                    target=process_audio_thread,
                    args=(simulator, simulator.audio_queue, simulator.processed_audio_queue, st.session_state.stop_event),
                    daemon=True
                )
                st.session_state.audio_thread.start()
                st.session_state.process_thread.start()
                st.success("Audio started!")
            else:
                # Stop audio
                st.session_state.audio_running = False
                st.session_state.stop_event.set()
                st.success("Audio stopped!")
        
        st.session_state.audio_running = st.session_state.audio_running and not st.session_state.stop_event.is_set()
        
        # Determinism controls
        st.subheader("🎲 Determinism")
        seed = st.number_input("Seed", value=12345, step=1)
        if st.button("Set Seed"):
            simulator.set_seed(int(seed))
            st.success(f"Seed set to {seed}")
        
        # Recording controls
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔴 Record"):
                simulator.recorder.start()
                st.success("Recording started!")
        with col2:
            if st.button("⏹️ Stop"):
                simulator.recorder.stop()
                st.success(f"Recording stopped! {len(simulator.recorder.frames)} frames recorded")
        
        if st.button("▶️ Replay"):
            if len(simulator.recorder.frames) > 0:
                simulator.mode = SimulationMode.REPLAY
                simulator.recorder.reset_replay()
                st.success("Replay started!")
            else:
                st.error("No recorded frames available!")
        
        # Physics controls
        st.subheader("⚛️ Physics")
        physics_config = {
            'blend_lorenz': st.slider("Blend Lorenz", 0.0, 1.0, 0.7, 0.05),
            'grav_enabled': st.checkbox("Enable Gravity", False),
            'dm_enabled': st.checkbox("Enable Dark Matter", False),
            'epsilon': st.slider("Epsilon (ε)", 0.01, 1.0, 0.1, 0.01),
            'rcutoff': st.slider("Cutoff Radius", 1.0, 50.0, 10.0, 0.5)
        }
        
        # Adaptive state controls
        st.subheader("🧠 Adaptive State")
        adapt_config = {
            'k': st.slider("k (Coupling)", 0.0, 2.0, 0.5, 0.1),
            'gamma': st.slider("γ (Decay)", 0.0, 1.0, 0.2, 0.05),
            'alpha': st.slider("α (Memory)", 0.0, 1.0, 0.3, 0.05),
            'sigma_sim': st.slider("σ (Similarity)", 0.1, 1.0, 0.3, 0.05)
        }
        
        # Synchronization controls
        st.subheader("🔄 Synchronization")
        sync_config = {
            'ksync': st.slider("K_sync", 0.0, 1.0, 0.1, 0.01)
        }
        
        # Timestep controls
        st.subheader("⏱️ Timestep")
        timestep_config = {
            'dt_max': st.slider("Max dt", 0.001, 0.1, 0.01, 0.001),
            'adaptive': st.checkbox("Adaptive Timestep", True)
        }
        
        # Dark matter controls
        st.subheader("🌌 Dark Matter")
        dm_params = {
            'rho0': st.slider("ρ₀", 0.1, 10.0, 1.0, 0.1),
            'rs': st.slider("r_s", 1.0, 20.0, 5.0, 0.5)
        }
        
        # Particle controls
        st.subheader("⚛️ Particles")
        if st.button("➕ Add Particle"):
            simulator.particles.append(Particle(
                np.random.random() * 10 - 5,
                np.random.random() * 10 - 5,
                np.random.random() * 10 - 5,
                440.0 + np.random.random() * 200
            ))
            st.success(f"Added particle! Total: {len(simulator.particles)}")
        
        if st.button("🗑️ Clear Particles"):
            simulator.particles = []
            st.success("Particles cleared!")
        
        # Update simulator with UI values
        update_simulator_from_ui(simulator, physics_config, adapt_config, sync_config, timestep_config, dm_params, audio_sensitivity)
    
    # Main content area
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Particles", len(simulator.particles))
    with col2:
        st.metric("Tokens", len(simulator.token_stream.tokens))
    with col3:
        token_rate = simulator.token_stream.update_rate()
        st.metric("Token Rate", f"{token_rate:.1f} tokens/sec")
    with col4:
        audio_energy = getattr(simulator, 'current_audio_energy', 0.0)
        st.metric("Audio Energy", f"{audio_energy:.3f}" if audio_energy > 0 else "0.000")
    with col5:
        st.metric("Recording", f"{len(simulator.recorder.frames)} frames" if simulator.recorder.recording else "Stopped")
    
    # Run simulation - CST v2.0 additive: Real-time audio-reactive processing
    if st.session_state.audio_running or len(simulator.particles) > 0:
        # Process audio frames continuously (limit to prevent blocking)
        frames_processed = 0
        max_frames_per_tick = 10
        while not simulator.processed_audio_queue.empty() and frames_processed < max_frames_per_tick:
            try:
                frame = simulator.processed_audio_queue.get_nowait()
                simulator._process_audio_frame(frame)
                frames_processed += 1
            except queue.Empty:
                break
        
        # Run simulation tick - particles respond to audio in real-time
        if len(simulator.particles) > 0:
            simulator.tick()
        
        # Update history - real-time metrics
        now = time.time()
        if now - st.session_state.last_update > 0.05:  # Update every 50ms for smoother real-time response
            psi = simulator.compute_psi()
            sync = simulator.compute_synchronization_metric()
            cons = simulator.compute_conservation_stats()
            
            st.session_state.history['time'].append(now)
            st.session_state.history['psi_total'].append(psi['psiTotal'])
            st.session_state.history['sync_r'].append(sync['r'])
            st.session_state.history['energy'].append(cons['Etotal'])
            st.session_state.history['token_rate'].append(token_rate)
            st.session_state.history['particle_count'].append(len(simulator.particles))
            
            # Keep only last 200 points
            max_points = 200
            for key in st.session_state.history:
                if len(st.session_state.history[key]) > max_points:
                    st.session_state.history[key] = st.session_state.history[key][-max_points:]
            
            st.session_state.last_update = now
    
    # Visualization tabs - CST v2.0 additive: Continuous streaming updates
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["3D Visualization", "Metrics", "Psi Breakdown", "Tokens", "Export"])
    
    with tab1:
        # Create placeholder for continuous updates (like a live TV show)
        plot_placeholder = st.empty()
        if len(simulator.particles) > 0:
            fig_3d = create_3d_scatter(simulator.particles)
            plot_placeholder.plotly_chart(fig_3d, use_container_width=True, key="live_3d_plot")
        else:
            plot_placeholder.info("Add particles to see 3D visualization")
    
    with tab2:
        # Create placeholder for continuous streaming metrics (like a heart monitor)
        metrics_placeholder = st.empty()
        if len(st.session_state.history['time']) > 0:
            df = pd.DataFrame(st.session_state.history)
            
            fig = make_subplots(
                rows=3, cols=2,
                subplot_titles=("Psi Total", "Synchronization (r)", "Energy", "Token Rate", "Particle Count"),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}],
                       [{"colspan": 2}, None]]
            )
            
            # Psi total - streaming line
            fig.add_trace(
                go.Scatter(x=df['time'], y=df['psi_total'], name="Psi Total", 
                          line=dict(color='#00d4ff', width=2), mode='lines'),
                row=1, col=1
            )
            
            # Sync r - streaming line
            fig.add_trace(
                go.Scatter(x=df['time'], y=df['sync_r'], name="Sync r", 
                          line=dict(color='#7b2ff7', width=2), mode='lines'),
                row=1, col=2
            )
            
            # Energy - streaming line
            fig.add_trace(
                go.Scatter(x=df['time'], y=df['energy'], name="Energy", 
                          line=dict(color='#f06eaa', width=2), mode='lines'),
                row=2, col=1
            )
            
            # Token rate - streaming line
            fig.add_trace(
                go.Scatter(x=df['time'], y=df['token_rate'], name="Token Rate", 
                          line=dict(color='#00ff00', width=2), mode='lines'),
                row=2, col=2
            )
            
            # Particle count - streaming line
            fig.add_trace(
                go.Scatter(x=df['time'], y=df['particle_count'], name="Particle Count", 
                          line=dict(color='#ff8c00', width=2), mode='lines'),
                row=3, col=1
            )
            
            # Update layout for continuous streaming feel
            fig.update_layout(
                height=900, 
                showlegend=False,
                template='plotly_dark',
                xaxis_rangeslider_visible=False
            )
            
            # Update axes for streaming effect
            for i in range(1, 6):
                row = (i-1)//2+1 if i <= 4 else 3
                col = (i-1)%2+1 if i <= 4 else 1
                fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)', row=row, col=col)
                fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)', row=row, col=col)
            
            metrics_placeholder.plotly_chart(fig, use_container_width=True, key="live_metrics_plot")
        else:
            metrics_placeholder.info("Start simulation to see metrics")
    
    with tab3:
        if len(simulator.particles) > 0:
            psi = simulator.compute_psi()
            sync = simulator.compute_synchronization_metric()
            cons = simulator.compute_conservation_stats()
            virial = simulator.compute_virial()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Psi Breakdown")
                st.metric("Energy Term", f"{psi['terms']['energyTerm']:.3f}")
                st.metric("Lambda Term", f"{psi['terms']['lambdaTerm']:.3f}")
                st.metric("Velocity Integral", f"{psi['terms']['velocityIntegralTerm']:.3f}")
                st.metric("x12 Integral", f"{psi['terms']['x12IntegralTerm']:.3f}")
                st.metric("Omega Term", f"{psi['terms']['omegaTerm']:.3f}")
                st.metric("Potential Term", f"{psi['terms']['potentialTerm']:.3f}")
                st.metric("**Total Psi**", f"**{psi['psiTotal']:.3f}**")
            
            with col2:
                st.subheader("Synchronization")
                st.metric("Order Parameter (r)", f"{sync['r']:.3f}")
                st.metric("Mean Theta", f"{sync['meanTheta']:.3f} rad")
                
                st.subheader("Conservation")
                st.metric("Total Energy", f"{cons['Etotal']:.3e}")
                st.metric("Energy Drift", f"{cons['drift']['E']*100:.2f}%")
                st.metric("Momentum |P|", f"{np.linalg.norm(cons['P']):.3e}")
                st.metric("Angular Momentum |L|", f"{np.linalg.norm(cons['L']):.3e}")
                st.metric("Virial Ratio", f"{virial['ratio']:.3f} {'✓' if virial['ok'] else '✗'}")
        else:
            st.info("Add particles to see diagnostics")
    
    with tab4:
        st.subheader("Token Stream")
        
        if len(simulator.token_stream.tokens) > 0:
            # Show recent tokens
            recent_tokens = simulator.token_stream.tokens[-50:]
            for token in reversed(recent_tokens):
                if token.get('type') == 'audio_frame':
                    st.json({
                        "Type": "Audio Frame",
                        "RMS": token.get('rmsEnergy', 0),
                        "Centroid": token.get('spectralCentroid', 0),
                        "Frequencies": len(token.get('topFrequencies', []))
                    })
                elif token.get('type') == 'phi_harmonic':
                    st.json({
                        "Type": "Phi Harmonic",
                        "Frequency": token.get('harmonic', 0),
                        "Index": token.get('harmonicIndex', 0)
                    })
        else:
            st.info("No tokens generated yet. Start audio to generate tokens.")
    
    with tab5:
        st.subheader("Export Tokens")
        
        # Manual refresh and pause controls for export tab
        col_refresh1, col_refresh2 = st.columns(2)
        with col_refresh1:
            if st.button("🔄 Refresh Data", key="refresh_export"):
                st.rerun()
        with col_refresh2:
            pause_state = st.session_state.get('auto_refresh_paused', False)
            button_label = "▶️ Resume Auto-Refresh" if pause_state else "⏸️ Pause Auto-Refresh"
            if st.button(button_label, key="pause_refresh"):
                st.session_state.auto_refresh_paused = not pause_state
                st.rerun()
        
        if st.session_state.get('auto_refresh_paused', False):
            st.info("⏸️ Auto-refresh paused. Click '🔄 Refresh Data' to update.")
        
        if len(simulator.token_stream.tokens) > 0:
            # Prepare metadata
            metadata = {
                "exportDate": datetime.now().isoformat(),
                "totalTokens": len(simulator.token_stream.tokens),
                "tokenGenerationRate": f"{simulator.token_stream.count_per_sec:.2f} tokens/sec",
                "engine": "12D Cosmic Synapse Theory",
                "version": "2.0",
                "mode": simulator.mode.value,
                "seed": simulator.seed,
                "particleCount": len(simulator.particles),
                "physics": {
                    "blendLorenz": simulator.physics.blendLorenz,
                    "gravEnabled": simulator.physics.gravEnabled,
                    "dmEnabled": simulator.physics.dmEnabled
                }
            }
            
            # Create export data
            export_data = {
                "metadata": metadata,
                "tokens": simulator.token_stream.tokens
            }
            
            # Convert to JSON string
            json_str = json.dumps(export_data, indent=2)
            
            # Download button with unique key to prevent 404 errors
            download_key = f"download_tokens_{len(simulator.token_stream.tokens)}"
            st.download_button(
                label="💾 Download Tokens (JSON)",
                data=json_str.encode('utf-8'),
                file_name=f"cosmic_tokens_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                key=download_key
            )
            
            st.text_area("Preview (first 1000 chars)", json_str[:1000] + "..." if len(json_str) > 1000 else json_str, height=200, key="token_preview")
            
            # Export recording
            if len(simulator.recorder.frames) > 0:
                st.subheader("Export Recording")
                recording_data = [
                    {
                        "timestamp": f.timestamp,
                        "rmsEnergy": f.rmsEnergy,
                        "frequencyData": f.frequencyData,
                        "spectralCentroid": f.spectralCentroid,
                        "harmonics": f.harmonics
                    }
                    for f in simulator.recorder.frames
                ]
                recording_json = json.dumps(recording_data, indent=2)
                
                recording_key = f"download_recording_{len(simulator.recorder.frames)}"
                st.download_button(
                    label="💾 Download Recording (JSON)",
                    data=recording_json.encode('utf-8'),
                    file_name=f"cosmic_recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    key=recording_key
                )
        else:
            st.info("No tokens to export yet. Start audio to generate tokens.")
    
    # Auto-refresh mechanism - CST v2.0 additive: Continuous streaming updates (like live TV/heart monitor)
    # Disable auto-refresh if paused
    auto_refresh_paused = st.session_state.get('auto_refresh_paused', False)
    
    if (st.session_state.audio_running or len(simulator.particles) > 0) and not auto_refresh_paused:
        # Use a more controlled refresh approach to prevent download button issues
        if 'last_rerun' not in st.session_state:
            st.session_state.last_rerun = time.time()
        
        # Refresh every 0.1 seconds for smooth continuous streaming (10 FPS UI update)
        # This provides smooth real-time updates like watching a live feed or heart monitor
        if time.time() - st.session_state.last_rerun > 0.1:
            st.session_state.last_rerun = time.time()
            time.sleep(0.02)  # Small delay to prevent excessive CPU usage
            st.rerun()


if __name__ == "__main__":
    main()

