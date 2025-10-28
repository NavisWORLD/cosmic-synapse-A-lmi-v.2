using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace CosmicSynapse
{
    /// <summary>
    /// Main manager for the Cosmic Synapse simulation.
    /// Manages particles, forces, and simulation state.
    /// </summary>
    public class CosmosManager : MonoBehaviour
    {
        [Header("Simulation Parameters")]
        [SerializeField] private float omega = 0.5f;          // Golden ratio frequency (φ)
        [SerializeField] private float lambda = 0.1f;         // Lyapunov exponent
        [SerializeField] private float damping = 0.98f;       // Velocity damping
        [SerializeField] private float timeScale = 1.0f;      // Time scale multiplier
        
        [Header("Particle Settings")]
        [SerializeField] private int particleCount = 1000;
        [SerializeField] private float particleSize = 0.05f;
        [SerializeField] private bool show3D = false;
        
        [Header("Physics")]
        [SerializeField] private float gravityStrength = 0.0001f;
        [SerializeField] private float swirlStrength = 0.5f;
        [SerializeField] private float bowlStrength = 0.01f;
        [SerializeField] private float softening = 0.1f;
        
        private ParticleSystem particleSystem;
        private AudioManager audioManager;
        private FFTAnalyzer fftAnalyzer;
        private ForceCalculator forceCalculator;
        private IPCBridgeClient ipcClient;
        
        private List<Particle> particles = new List<Particle>();
        private float simulationTime = 0f;
        private bool isSimulating = false;
        
        private void Awake()
        {
            // Initialize components
            particleSystem = GetComponent<ParticleSystem>();
            audioManager = FindObjectOfType<AudioManager>();
            fftAnalyzer = GetComponent<FFTAnalyzer>();
            forceCalculator = GetComponent<ForceCalculator>();
            ipcClient = GetComponent<IPCBridgeClient>();
        }
        
        private void Start()
        {
            InitializeParticles();
            StartSimulation();
            
            // Connect to IPC bridge
            if (ipcClient != null)
            {
                ipcClient.ConnectToBridge();
            }
        }
        
        private void InitializeParticles()
        {
            particles.Clear();
            
            // Initialize particles in a ring structure
            float goldenAngle = 2.39996322f * Mathf.PI / 180f; // Golden angle in radians
            
            for (int i = 0; i < particleCount; i++)
            {
                float angle = goldenAngle * i;
                float radius = Mathf.Sqrt(i) * 0.1f;
                
                Vector3 position = new Vector3(
                    Mathf.Cos(angle) * radius,
                    Mathf.Sin(angle) * radius,
                    show3D ? (Mathf.Pow(i, 0.5f) * 0.05f) : 0f
                );
                
                Particle particle = new Particle
                {
                    position = position,
                    velocity = Vector3.zero,
                    age = Random.Range(0f, 100f)
                };
                
                particles.Add(particle);
            }
            
            Debug.Log($"Initialized {particles.Count} particles");
        }
        
        private void Update()
        {
            if (!isSimulating) return;
            
            float deltaTime = Time.deltaTime * timeScale;
            simulationTime += deltaTime;
            
            // Get audio data
            float[] audioSpectrum = fftAnalyzer?.GetSpectrum() ?? new float[64];
            
            // Update particles
            UpdateParticles(deltaTime, audioSpectrum);
            
            // Apply forces
            ApplyForces(deltaTime);
            
            // Spawn masses based on audio amplitude
            if (audioManager != null && audioManager.GetAverageAmplitude() > 0.5f)
            {
                SpawnMassAtCenter();
            }
        }
        
        private void UpdateParticles(float deltaTime, float[] audioSpectrum)
        {
            for (int i = 0; i < particles.Count; i++)
            {
                Particle p = particles[i];
                
                // Update position
                p.position += p.velocity * deltaTime;
                
                // Update age
                p.age += deltaTime;
                
                // Stochastic resonance from audio
                if (audioSpectrum != null && audioSpectrum.Length > 0)
                {
                    int freqBin = Mathf.FloorToInt(p.age % audioSpectrum.Length);
                    float noise = audioSpectrum[freqBin] * lambda;
                    p.velocity += Random.insideUnitSphere * noise;
                }
                
                particles[i] = p;
            }
        }
        
        private void ApplyForces(float deltaTime)
        {
            for (int i = 0; i < particles.Count; i++)
            {
                Particle p = particles[i];
                Vector3 force = Vector3.zero;
                
                // Conservative bowl potential
                Vector3 center = Vector3.zero;
                Vector3 toCenter = center - p.position;
                float dist = toCenter.magnitude;
                force += toCenter.normalized * bowlStrength * dist;
                
                // Swirl restraint (perpendicular to radius)
                Vector3 perpendicular = new Vector3(-toCenter.y, toCenter.x, 0);
                force += perpendicular.normalized * swirlStrength * omega;
                
                // Velocity damping
                p.velocity *= damping;
                
                // Apply force
                p.velocity += force * deltaTime;
                
                particles[i] = p;
            }
        }
        
        private void StartSimulation()
        {
            isSimulating = true;
            Debug.Log("Cosmic Synapse simulation started");
        }
        
        public void StopSimulation()
        {
            isSimulating = false;
            Debug.Log("Cosmic Synapse simulation stopped");
        }
        
        public void SpawnMassAtCenter()
        {
            // Create a new "black hole" or "star" mass at center
            GameObject mass = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            mass.transform.position = Vector3.zero;
            mass.transform.localScale = Vector3.one * 0.2f;
            mass.name = "Mass Limb";
            
            // Add gravitational influence component
            MassInfluence influence = mass.AddComponent<MassInfluence>();
            influence.gravityStrength = gravityStrength * 10f;
            
            // Destroy after some time
            Destroy(mass, 10f);
        }
        
        public List<Particle> GetParticles()
        {
            return particles;
        }
        
        // Property setters for UI
        public void SetOmega(float value) { omega = value; }
        public void SetLambda(float value) { lambda = value; }
        public void SetDamping(float value) { damping = value; }
        public void SetTimeScale(float value) { timeScale = value; }
        public void SetParticleCount(int value)
        {
            particleCount = value;
            InitializeParticles();
        }
    }
    
    /// <summary>
    /// Particle data structure
    /// </summary>
    [System.Serializable]
    public struct Particle
    {
        public Vector3 position;
        public Vector3 velocity;
        public float age;
    }
}

