using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace CosmicSynapse
{
    /// <summary>
    /// Calculates gravitational and attractive forces for particles.
    /// Implements N-body force calculation with softening parameter.
    /// </summary>
    public class ForceCalculator : MonoBehaviour
    {
        [Header("Force Parameters")]
        [SerializeField] private float gravityConstant = 0.0001f;
        [SerializeField] private float softeningParameter = 0.1f;
        [SerializeField] private float maxForce = 100f;
        
        /// <summary>
        /// Calculate gravitational force between two particles.
        /// </summary>
        public Vector3 CalculateGravitationalForce(Vector3 pos1, Vector3 pos2, float mass1, float mass2)
        {
            Vector3 direction = pos2 - pos1;
            float distance = direction.magnitude;
            
            // Softened potential to prevent singularities
            float softenedDistance = distance + softeningParameter;
            
            // Newtonian gravity: F = G * m1 * m2 / r²
            float forceMagnitude = gravityConstant * mass1 * mass2 / (softenedDistance * softenedDistance);
            
            Vector3 force = direction.normalized * forceMagnitude;
            
            // Limit force magnitude
            if (force.magnitude > maxForce)
            {
                force = force.normalized * maxForce;
            }
            
            return force;
        }
        
        /// <summary>
        /// Calculate force towards center (conservative bowl potential)
        /// </summary>
        public Vector3 CalculateBowlForce(Vector3 position, Vector3 center, float bowlStrength)
        {
            Vector3 toCenter = center - position;
            return toCenter * bowlStrength;
        }
        
        /// <summary>
        /// Calculate swirl force (perpendicular to radius).
        /// Creates rotational motion around center.
        /// </summary>
        public Vector3 CalculateSwirlForce(Vector3 position, float omega)
        {
            // Get perpendicular vector to radius (in XY plane)
            Vector3 perpendicular = new Vector3(-position.y, position.x, 0);
            
            // Swirl strength proportional to omega
            return perpendicular.normalized * omega;
        }
        
        /// <summary>
        /// Calculate stochastic noise force modulated by audio.
        /// </summary>
        public Vector3 CalculateStochasticForce(float[] audioSpectrum, int frequencyBin, float lambda)
        {
            if (audioSpectrum == null || frequencyBin >= audioSpectrum.Length)
            {
                return Vector3.zero;
            }
            
            float intensity = audioSpectrum[frequencyBin] * lambda;
            return Random.insideUnitSphere * intensity;
        }
        
        /// <summary>
        /// Calculate damping force to prevent infinite energy.
        /// </summary>
        public Vector3 CalculateDampingForce(Vector3 velocity, float dampingFactor)
        {
            return -velocity * dampingFactor;
        }
    }
}

