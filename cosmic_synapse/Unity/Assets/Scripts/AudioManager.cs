using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace CosmicSynapse
{
    /// <summary>
    /// Manages audio input and analysis for stochastic resonance.
    /// </summary>
    public class AudioManager : MonoBehaviour
    {
        [Header("Audio Settings")]
        [SerializeField] private int sampleRate = 44100;
        [SerializeField] private int bufferSize = 512;
        [SerializeField] private string microphoneDevice = "";
        
        [Header("Amplitude Thresholds")]
        [SerializeField] private float triggerThreshold = 0.5f;
        
        private AudioClip microphoneClip;
        private bool isRecording = false;
        private float averageAmplitude = 0f;
        private float[] audioSamples;
        
        private void Start()
        {
            InitializeAudio();
        }
        
        private void InitializeAudio()
        {
            // Get microphone devices
            string[] devices = Microphone.devices;
            
            if (devices.Length == 0)
            {
                Debug.LogWarning("No microphone devices found");
                return;
            }
            
            if (string.IsNullOrEmpty(microphoneDevice))
            {
                microphoneDevice = devices[0];
            }
            
            Debug.Log($"Using microphone: {microphoneDevice}");
            
            audioSamples = new float[bufferSize];
        }
        
        public void StartRecording()
        {
            if (isRecording) return;
            
            int minFreq, maxFreq;
            Microphone.GetDeviceCaps(microphoneDevice, out minFreq, out maxFreq);
            
            int selectedFreq = Mathf.Clamp(sampleRate, minFreq, maxFreq);
            
            microphoneClip = Microphone.Start(microphoneDevice, true, 1, selectedFreq);
            isRecording = true;
            
            Debug.Log("Recording started");
        }
        
        public void StopRecording()
        {
            if (!isRecording) return;
            
            Microphone.End(microphoneDevice);
            isRecording = false;
            
            Debug.Log("Recording stopped");
        }
        
        private void Update()
        {
            if (isRecording && microphoneClip != null)
            {
                UpdateAudioData();
            }
        }
        
        private void UpdateAudioData()
        {
            int currentSample = Microphone.GetPosition(microphoneDevice);
            
            if (currentSample < bufferSize) return;
            
            // Get audio data
            microphoneClip.GetData(audioSamples, currentSample - bufferSize);
            
            // Calculate average amplitude
            float sum = 0f;
            for (int i = 0; i < bufferSize; i++)
            {
                sum += Mathf.Abs(audioSamples[i]);
            }
            
            averageAmplitude = sum / bufferSize;
        }
        
        public float GetAverageAmplitude()
        {
            return averageAmplitude;
        }
        
        public float[] GetAudioSamples()
        {
            return audioSamples;
        }
        
        public bool IsRecording()
        {
            return isRecording;
        }
        
        public bool ShouldTrigger()
        {
            return averageAmplitude > triggerThreshold;
        }
    }
}

