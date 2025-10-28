using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace CosmicSynapse
{
    /// <summary>
    /// Component that adds gravitational influence to spawned masses.
    /// </summary>
    public class MassInfluence : MonoBehaviour
    {
        public float gravityStrength = 1f;
        public float massRadius = 0.5f;
        
        public Vector3 GetGravitationalForce(Vector3 position, float particleMass)
        {
            Vector3 direction = transform.position - position;
            float distance = direction.magnitude;
            
            if (distance < 0.01f) return Vector3.zero;
            
            float forceMagnitude = gravityStrength * particleMass / (distance * distance);
            return direction.normalized * forceMagnitude;
        }
    }
}

