using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using TMPro;

namespace CosmicSynapse
{
    /// <summary>
    /// Manages the Unity UI for controlling simulation parameters.
    /// </summary>
    public class UIManager : MonoBehaviour
    {
        [Header("UI References")]
        [SerializeField] private TMP_Text timeScaleText;
        [SerializeField] private TMP_Text particleCountText;
        [SerializeField] private TMP_Text statisticsText;
        
        [Header("Sliders")]
        [SerializeField] private Slider omegaSlider;
        [SerializeField] private Slider lambdaSlider;
        [SerializeField] private Slider dampingSlider;
        [SerializeField] private Slider timeScaleSlider;
        
        [Header("Buttons")]
        [SerializeField] private Button startButton;
        [SerializeField] private Button stopButton;
        [SerializeField] private Button spawnButton;
        [SerializeField] private Toggle microphoneToggle;
        
        private CosmosManager cosmosManager;
        private AudioManager audioManager;
        
        private void Start()
        {
            cosmosManager = FindObjectOfType<CosmosManager>();
            audioManager = FindObjectOfType<AudioManager>();
            
            SetupUI();
        }
        
        private void SetupUI()
        {
            // Sliders
            omegaSlider.value = 0.5f;
            omegaSlider.onValueChanged.AddListener(OnOmegaChanged);
            
            lambdaSlider.value = 0.1f;
            lambdaSlider.onValueChanged.AddListener(OnLambdaChanged);
            
            dampingSlider.value = 0.98f;
            dampingSlider.onValueChanged.AddListener(OnDampingChanged);
            
            timeScaleSlider.value = 1.0f;
            timeScaleSlider.onValueChanged.AddListener(OnTimeScaleChanged);
            
            // Buttons
            startButton.onClick.AddListener(OnStartClicked);
            stopButton.onClick.AddListener(OnStopClicked);
            spawnButton.onClick.AddListener(OnSpawnClicked);
            
            microphoneToggle.onValueChanged.AddListener(OnMicrophoneToggled);
        }
        
        private void OnOmegaChanged(float value)
        {
            cosmosManager?.SetOmega(value);
            UpdateTimeScaleText();
        }
        
        private void OnLambdaChanged(float value)
        {
            cosmosManager?.SetLambda(value);
        }
        
        private void OnDampingChanged(float value)
        {
            cosmosManager?.SetDamping(value);
        }
        
        private void OnTimeScaleChanged(float value)
        {
            cosmosManager?.SetTimeScale(value);
            UpdateTimeScaleText();
        }
        
        private void OnStartClicked()
        {
            cosmosManager?.StartSimulation();
        }
        
        private void OnStopClicked()
        {
            cosmosManager?.StopSimulation();
        }
        
        private void OnSpawnClicked()
        {
            cosmosManager?.SpawnMassAtCenter();
        }
        
        private void OnMicrophoneToggled(bool value)
        {
            if (value)
            {
                audioManager?.StartRecording();
            }
            else
            {
                audioManager?.StopRecording();
            }
        }
        
        private void Update()
        {
            UpdateStatistics();
        }
        
        private void UpdateStatistics()
        {
            if (statisticsText == null) return;
            
            float amplitude = audioManager?.GetAverageAmplitude() ?? 0f;
            bool recording = audioManager?.IsRecording() ?? false;
            
            statisticsText.text = $"Amplitude: {amplitude:F3}\nRecording: {(recording ? "ON" : "OFF")}";
        }
        
        private void UpdateTimeScaleText()
        {
            if (timeScaleText != null && timeScaleSlider != null)
            {
                timeScaleText.text = $"Time Scale: {timeScaleSlider.value:F2}x";
            }
        }
    }
}

