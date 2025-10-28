using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace CosmicSynapse
{
    /// <summary>
    /// Performs FFT standardization for audio visualization and stochastic resonance.
    /// Based on: "FFT standardization for multimodal audio-visual emotional analyses in the interactive console Cyberdam/N."
    /// </summary>
    public class FFTAnalyzer : MonoBehaviour
    {
        [Header("FFT Settings")]
        [SerializeField] private int fftSize = 512;
        [SerializeField] private WindowFunction windowFunction = WindowFunction.Hanning;
        
        private AudioSource audioSource;
        private float[] spectrum;
        private float[] psd; // Power Spectral Density
        
        private void Start()
        {
            audioSource = GetComponent<AudioSource>();
            spectrum = new float[fftSize];
            psd = new float[fftSize / 2];
        }
        
        private void Update()
        {
            if (audioSource != null && audioSource.isPlaying)
            {
                AnalyzeAudio();
            }
        }
        
        private void AnalyzeAudio()
        {
            // Get spectrum data
            audioSource.GetSpectrumData(spectrum, 0, FFTWindow.BlackmanHarris);
            
            // Compute PSD
            ComputePSD();
        }
        
        private void ComputePSD()
        {
            // PSD = |X(f)|²
            // Normalize by frequency bin width
            float binWidth = AudioSettings.outputSampleRate / (2f * fftSize);
            
            for (int i = 0; i < psd.Length; i++)
            {
                float magnitude = spectrum[i];
                psd[i] = magnitude * magnitude / binWidth;
                
                // Apply windowing correction if needed
                psd[i] *= GetWindowCorrection();
            }
        }
        
        private float GetWindowCorrection()
        {
            // Window correction factor based on window function
            switch (windowFunction)
            {
                case WindowFunction.Hanning:
                    return 1.5f;
                case WindowFunction.Hamming:
                    return 1.36f;
                case WindowFunction.Blackman:
                    return 1.73f;
                default:
                    return 1.0f;
            }
        }
        
        public float[] GetSpectrum()
        {
            return spectrum;
        }
        
        public float[] GetPSD()
        {
            return psd;
        }
        
        public float GetPowerAtFrequency(float frequency)
        {
            // Convert frequency to bin index
            int binIndex = Mathf.FloorToInt(frequency * fftSize / AudioSettings.outputSampleRate);
            
            if (binIndex >= 0 && binIndex < psd.Length)
            {
                return psd[binIndex];
            }
            
            return 0f;
        }
    }
    
    public enum WindowFunction
    {
        None,
        Hanning,
        Hamming,
        Blackman,
        BlackmanHarris
    }
}

