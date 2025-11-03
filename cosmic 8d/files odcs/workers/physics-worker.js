// Physics Processing Web Worker
// Handles force calculations and particle updates off the main thread

self.addEventListener('message', (e) => {
    const { type, data } = e.data;
    
    switch(type) {
        case 'calculateForces':
            calculateForces(data);
            break;
        case 'calculateOmega':
            calculateOmega(data);
            break;
        case 'updateParticles':
            updateParticles(data);
            break;
        case 'init':
            initializeWorker(data);
            break;
    }
});

let constants = {
    G: 6.674e-11,
    a0: 5.29177210903e-11,
    m0: 9.1093837015e-31
};

function initializeWorker(config) {
    if (config.constants) {
        constants = { ...constants, ...config.constants };
    }
    self.postMessage({ type: 'initialized', success: true });
}

function calculateForces(data) {
    const { particles, dt, interactionRadius } = data;
    const forces = new Float32Array(particles.length * 3);
    
    for (let i = 0; i < particles.length; i++) {
        const p1 = particles[i];
        const fx = forces[i * 3];
        const fy = forces[i * 3 + 1];
        const fz = forces[i * 3 + 2];
        let forceX = fx || 0;
        let forceY = fy || 0;
        let forceZ = fz || 0;
        
        for (let j = 0; j < particles.length; j++) {
            if (i === j) continue;
            
            const p2 = particles[j];
            const dx = p2.position[0] - p1.position[0];
            const dy = p2.position[1] - p1.position[1];
            const dz = p2.position[2] - p1.position[2];
            const r = Math.sqrt(dx * dx + dy * dy + dz * dz + 1e-10);
            
            if (r < interactionRadius) {
                // Gravitational force
                const F = constants.G * p1.mass * p2.mass / (r * r);
                forceX += F * dx / r;
                forceY += F * dy / r;
                forceZ += F * dz / r;
            }
        }
        
        forces[i * 3] = forceX;
        forces[i * 3 + 1] = forceY;
        forces[i * 3 + 2] = forceZ;
    }
    
    self.postMessage({
        type: 'forcesResult',
        data: { forces: Array.from(forces), particleCount: particles.length }
    });
}

function calculateOmega(data) {
    const { particles, interactionRadius, sigma } = data;
    const omegaResults = new Float32Array(particles.length);
    const connections = [];
    
    for (let i = 0; i < particles.length; i++) {
        const p1 = particles[i];
        let omega = 0;
        const particleConnections = [];
        
        for (let j = 0; j < particles.length; j++) {
            if (i === j) continue;
            
            const p2 = particles[j];
            const dx = p2.position[0] - p1.position[0];
            const dy = p2.position[1] - p1.position[1];
            const dz = p2.position[2] - p1.position[2];
            const r = Math.sqrt(dx * dx + dy * dy + dz * dz);
            
            if (r < interactionRadius) {
                // Complete synaptic strength formula: Ωᵢⱼ = [G·mᵢ·mⱼ/(r²ᵢⱼ·a₀·m₀)] · exp[-(x₁₂,ᵢ - x₁₂,ⱼ)²/(2σ²)]
                const r2 = r * r + 1e-10;
                const gravCoupling = (constants.G * p1.mass * p2.mass) / (r2 * constants.a0 * constants.m0);
                
                const stateDiff = p1.x12 - p2.x12;
                const similarity = Math.exp(-stateDiff * stateDiff / (2 * sigma * sigma));
                
                const omega_ij = gravCoupling * similarity;
                omega += omega_ij;
                
                particleConnections.push({
                    particleIndex: j,
                    strength: omega_ij
                });
            }
        }
        
        omegaResults[i] = omega;
        connections.push(particleConnections);
    }
    
    self.postMessage({
        type: 'omegaResult',
        data: {
            omega: Array.from(omegaResults),
            connections: connections
        }
    });
}

function updateParticles(data) {
    const { particles, forces, dt, k, gamma, alpha } = data;
    const updatedParticles = [];
    
    for (let i = 0; i < particles.length; i++) {
        const p = particles[i];
        const fx = forces[i * 3];
        const fy = forces[i * 3 + 1];
        const fz = forces[i * 3 + 2];
        
        // Update velocity
        const vx = p.velocity[0] + (fx / p.mass) * dt;
        const vy = p.velocity[1] + (fy / p.mass) * dt;
        const vz = p.velocity[2] + (fz / p.mass) * dt;
        
        // Update position
        const px = p.position[0] + vx * dt;
        const py = p.position[1] + vy * dt;
        const pz = p.position[2] + vz * dt;
        
        // Update 12th dimension: dx12/dt = k·Ω - γ·x12
        const dx12_dt = k * p.omega - gamma * p.x12;
        const newX12 = Math.max(-1, Math.min(1, p.x12 + dx12_dt * dt));
        
        // Update memory: dm12/dt = α(x12 - m12)
        const newMemory12 = p.memory12 + alpha * (newX12 - p.memory12) * dt;
        
        // Update integrated values
        const v2_sum = vx * vx + vy * vy + vz * vz;
        const integratedVelocityPath = p.integratedVelocityPath + v2_sum * dt;
        const integratedX12Change = p.integratedX12Change + Math.abs(dx12_dt) * dt;
        
        updatedParticles.push({
            index: i,
            position: [px, py, pz],
            velocity: [vx, vy, vz],
            x12: newX12,
            memory12: newMemory12,
            integratedVelocityPath: integratedVelocityPath,
            integratedX12Change: integratedX12Change
        });
    }
    
    self.postMessage({
        type: 'updateResult',
        data: { particles: updatedParticles }
    });
}

