// Particle System Module
// 12D Cosmic Synapse Theory - Particle12D Class
// Author: Cory Shane Davis

import { PHI, G, c, h, k_B, a0, m0 } from './constants.js';

export class Particle12D {
    constructor(id, position = null) {
        this.id = id;
        
        // Physical dimensions (1-11)
        this.position = position || new Float32Array([
            (Math.random() - 0.5) * 200,
            (Math.random() - 0.5) * 200,
            (Math.random() - 0.5) * 200
        ]);
        
        this.velocity = new Float32Array([
            (Math.random() - 0.5) * 2,
            (Math.random() - 0.5) * 2,
            (Math.random() - 0.5) * 2
        ]);
        
        this.mass = 1e20 + Math.random() * 1e20;
        this.energy = 0.5 * this.mass * (
            this.velocity[0]**2 + 
            this.velocity[1]**2 + 
            this.velocity[2]**2
        );
        
        // Higher dimensions
        this.entropy = Math.random() * 10;
        this.frequency = 440 + Math.random() * 880;
        this.phase = Math.random() * Math.PI * 2;
        
        // 12th dimension - Internal adaptive state
        this.x12 = (Math.random() - 0.5) * 2; // [-1, 1]
        this.memory12 = 0; // Memory of x12
        
        // Network properties
        this.omega = 0; // Synaptic strength
        this.connections = [];
        this.lyapunov = 0.1 + Math.random() * 0.1;
        
        // Additional dimensions (4-11) tracking
        this.dimensions = {
            dim4: this.velocity[0], // vx (dimension 4)
            dim5: this.velocity[1], // vy (dimension 5)
            dim6: this.velocity[2], // vz (dimension 6)
            dim7: 0, // Time dimension (tracked externally)
            dim8: 0, // Cosmic energy Eᶜ (calculated)
            dim9: 0, // Entropy S (calculated)
            dim10: 0, // Frequency ν (calculated)
            dim11: 0 // Connectivity phase Θ (calculated)
        };
        
        // Integration accumulators for proper time integration
        this.integratedVelocityPath = 0; // ∫₀ᵗ Σₖ₌₁¹¹(dxᵢ,ₖ/dt)² dt
        this.integratedX12Change = 0; // ∫₀ᵗ|dx₁₂,ᵢ/dt|dt
        
        // Cosmic energy (dimension 8) - separate from kinetic
        this.cosmicEnergy = 0;
        
        // 11D gravitational potential
        this.gravitationalPotential11D = 0;
        
        // Visualization
        this.color = new Float32Array(3);
        this.updateColor();
        
        // History for trail effect
        this.trail = [];
        this.maxTrailLength = 20;
    }
    
    updateColor() {
        // Color based on internal state x12
        const hue = (this.x12 + 1) * 180; // Map [-1,1] to [0,360]
        const rgb = this.hslToRgb(hue / 360, 0.8, 0.5 + this.energy / 1e30);
        this.color[0] = rgb[0];
        this.color[1] = rgb[1];
        this.color[2] = rgb[2];
    }
    
    hslToRgb(h, s, l) {
        let r, g, b;
        if (s === 0) {
            r = g = b = l;
        } else {
            const hue2rgb = (p, q, t) => {
                if (t < 0) t += 1;
                if (t > 1) t -= 1;
                if (t < 1/6) return p + (q - p) * 6 * t;
                if (t < 1/2) return q;
                if (t < 2/3) return p + (q - p) * (2/3 - t) * 6;
                return p;
            };
            const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
            const p = 2 * l - q;
            r = hue2rgb(p, q, h + 1/3);
            g = hue2rgb(p, q, h);
            b = hue2rgb(p, q, h - 1/3);
        }
        return [r, g, b];
    }
    
    computeOmega(particles, radius, sigma) {
        this.omega = 0;
        this.connections = [];
        
        for (let other of particles) {
            if (other.id === this.id) continue;
            
            const dx = other.position[0] - this.position[0];
            const dy = other.position[1] - this.position[1];
            const dz = other.position[2] - this.position[2];
            const r = Math.sqrt(dx*dx + dy*dy + dz*dz);
            
            if (r < radius) {
                // Complete synaptic strength formula: Ωᵢⱼ = [G·mᵢ·mⱼ/(r²ᵢⱼ·a₀·m₀)] · exp[-(x₁₂,ᵢ - x₁₂,ⱼ)²/(2σ²)]
                const r2 = r * r + 1e-10; // Softening
                const gravCoupling = (G * this.mass * other.mass) / (r2 * a0 * m0);
                
                // Internal state similarity (12th dimension)
                const stateDiff = this.x12 - other.x12;
                const similarity = Math.exp(-stateDiff*stateDiff / (2*sigma*sigma));
                
                const omega_ij = gravCoupling * similarity;
                this.omega += omega_ij;
                this.connections.push({
                    particle: other,
                    strength: omega_ij
                });
            }
        }
        
        return this.omega;
    }
    
    updateInternalState(k, gamma, dt) {
        // 12th dimension evolution: dx12/dt = k·Ω - γ·x12
        const dx12_dt = k * this.omega - gamma * this.x12;
        
        // Accumulate integrated x12 change: ∫₀ᵗ|dx₁₂,ᵢ/dt|dt
        this.integratedX12Change += Math.abs(dx12_dt) * dt;
        
        this.x12 += dx12_dt * dt;
        
        // Bound to [-1, 1]
        this.x12 = Math.max(-1, Math.min(1, this.x12));
    }
    
    updateMemory(alpha, dt) {
        // Memory tracks internal state: dm12/dt = α(x12 - m12)
        const dm12 = alpha * (this.x12 - this.memory12);
        this.memory12 += dm12 * dt;
    }
    
    applyForces(particles, dt) {
        const forces = [0, 0, 0];
        
        for (let other of particles) {
            if (other.id === this.id) continue;
            
            const dx = other.position[0] - this.position[0];
            const dy = other.position[1] - this.position[1];
            const dz = other.position[2] - this.position[2];
            const r = Math.sqrt(dx*dx + dy*dy + dz*dz + 1e-10);
            
            // Gravitational force
            const F = G * this.mass * other.mass / (r*r + 1e-10);
            
            forces[0] += F * dx / r;
            forces[1] += F * dy / r;
            forces[2] += F * dz / r;
        }
        
        // Connectivity force (based on synaptic strength)
        for (let conn of this.connections) {
            const other = conn.particle;
            const dx = other.position[0] - this.position[0];
            const dy = other.position[1] - this.position[1];
            const dz = other.position[2] - this.position[2];
            const r = Math.sqrt(dx*dx + dy*dy + dz*dz + 1e-10);
            
            const F = conn.strength * 0.001; // Scale factor
            
            forces[0] += F * dx / r;
            forces[1] += F * dy / r;
            forces[2] += F * dz / r;
        }
        
        // Update velocity
        this.velocity[0] += (forces[0] / this.mass) * dt;
        this.velocity[1] += (forces[1] / this.mass) * dt;
        this.velocity[2] += (forces[2] / this.mass) * dt;
        
        // Damping
        const damping = 0.99;
        this.velocity[0] *= damping;
        this.velocity[1] *= damping;
        this.velocity[2] *= damping;
    }
    
    updatePosition(dt) {
        // Accumulate integrated velocity path: ∫₀ᵗ Σₖ₌₁¹¹(dxᵢ,ₖ/dt)² dt
        const v2_sum = this.velocity[0]**2 + this.velocity[1]**2 + this.velocity[2]**2;
        this.integratedVelocityPath += v2_sum * dt;
        
        // Update dimensions tracking
        this.dimensions.dim4 = this.velocity[0];
        this.dimensions.dim5 = this.velocity[1];
        this.dimensions.dim6 = this.velocity[2];
        
        // Update trail
        this.trail.push([...this.position]);
        if (this.trail.length > this.maxTrailLength) {
            this.trail.shift();
        }
        
        // Update position
        this.position[0] += this.velocity[0] * dt;
        this.position[1] += this.velocity[1] * dt;
        this.position[2] += this.velocity[2] * dt;
        
        // Boundary conditions
        const boundary = 300;
        for (let i = 0; i < 3; i++) {
            if (Math.abs(this.position[i]) > boundary) {
                this.position[i] = Math.sign(this.position[i]) * boundary;
                this.velocity[i] *= -0.5;
            }
        }
    }
    
    updateEnergy(particles) {
        const v2 = this.velocity[0]**2 + this.velocity[1]**2 + this.velocity[2]**2;
        const kineticEnergy = 0.5 * this.mass * v2;
        
        // Internal state contribution (12th dimension)
        const internalEnergy = Math.abs(this.x12) * this.mass * c * c * 1e-10;
        
        // Connectivity contribution
        const connectivityEnergy = this.omega * kineticEnergy * 1e-10;
        
        // Cosmic energy (dimension 8): Eᶜ = kinetic + internal + connectivity
        this.cosmicEnergy = kineticEnergy + internalEnergy + connectivityEnergy;
        this.energy = this.cosmicEnergy;
        
        // Update dimension 8
        this.dimensions.dim8 = this.cosmicEnergy;
        
        // Calculate 11D gravitational potential: U_grav^11D = -Σⱼ≠ᵢ G·mᵢ·mⱼ/rᵢⱼ
        this.gravitationalPotential11D = 0;
        for (let other of particles) {
            if (other.id === this.id) continue;
            const dx = other.position[0] - this.position[0];
            const dy = other.position[1] - this.position[1];
            const dz = other.position[2] - this.position[2];
            const r = Math.sqrt(dx*dx + dy*dy + dz*dz + 1e-10);
            this.gravitationalPotential11D -= G * this.mass * other.mass / r;
        }
        
        // Update frequency (dimension 10) based on energy
        this.frequency = this.energy / h * 1e-20;
        this.dimensions.dim10 = this.frequency;
    }
    
    updateEntropy() {
        // S = k_B * ln(Ω + 1) - dimension 9
        if (this.omega > 0) {
            this.entropy = k_B * Math.log(this.omega + 1);
            this.dimensions.dim9 = this.entropy;
        } else {
            this.entropy = 0;
            this.dimensions.dim9 = 0;
        }
    }
    
    computeConnectivityPhase() {
        // Dimension 11: Connectivity phase Θ = arctan2(ΣΩ·sin(φ), ΣΩ·cos(φ))
        let sinSum = 0;
        let cosSum = 0;
        
        for (let conn of this.connections) {
            const phase = conn.particle.phase || 0;
            sinSum += conn.strength * Math.sin(phase);
            cosSum += conn.strength * Math.cos(phase);
        }
        
        this.dimensions.dim11 = Math.atan2(sinSum, cosSum);
        return this.dimensions.dim11;
    }
    
    computePsi() {
        // Complete 12D state function:
        // ψᵢ = (φ·Eᶜ,ᵢ)/c² + λ + ∫₀ᵗ Σₖ₌₁¹¹(dxᵢ,ₖ/dt)² dt + ∫₀ᵗ|dx₁₂,ᵢ/dt|dt + Ωᵢ·Eᶜ,ᵢ + U¹¹ᴰ_grav,i
        
        const term1 = PHI * this.cosmicEnergy / (c * c);
        const term2 = this.lyapunov;
        const term3 = this.integratedVelocityPath;
        const term4 = this.integratedX12Change;
        const term5 = this.omega * this.cosmicEnergy;
        const term6 = this.gravitationalPotential11D / (c * c);
        
        return term1 + term2 + term3 + term4 + term5 + term6;
    }
    
    applyAudioFrequency(frequency, magnitude) {
        // Map audio frequency to particle properties
        const freqRatio = frequency / 440; // A4 reference
        const harmonic = Math.pow(PHI, Math.floor(freqRatio));
        
        // Modulate internal state
        this.x12 += magnitude * 0.01 * Math.sin(harmonic * this.phase);
        this.x12 = Math.max(-1, Math.min(1, this.x12));
        
        // Modulate energy
        this.energy *= (1 + magnitude * 0.1 * harmonic);
        
        // Update phase
        this.phase += frequency * 0.001;
        if (this.phase > Math.PI * 2) this.phase -= Math.PI * 2;
        
        // Update visualization
        this.updateColor();
    }
}

