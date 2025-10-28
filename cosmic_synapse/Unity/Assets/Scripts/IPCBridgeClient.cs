using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Net.WebSockets;
using System.Text;
using System.Threading;
using System;

namespace CosmicSynapse
{
    /// <summary>
    /// Client for IPC communication with A-LMI system via WebSocket.
    /// </summary>
    public class IPCBridgeClient : MonoBehaviour
    {
        [Header("IPC Settings")]
        [SerializeField] private string serverUrl = "ws://localhost:8765";
        [SerializeField] private bool autoConnect = true;
        [SerializeField] private float reconnectInterval = 5f;
        
        private ClientWebSocket webSocket;
        private CancellationTokenSource cancellationTokenSource;
        private bool isConnected = false;
        private float lastReconnectAttempt = 0f;
        
        // Events
        public System.Action<string> OnCommandReceived;
        public System.Action<string> OnStatusSent;
        
        private void Start()
        {
            cancellationTokenSource = new CancellationTokenSource();
            
            if (autoConnect)
            {
                ConnectToBridge();
            }
        }
        
        private void Update()
        {
            if (!isConnected && autoConnect)
            {
                // Attempt reconnection
                if (Time.time - lastReconnectAttempt > reconnectInterval)
                {
                    ConnectToBridge();
                    lastReconnectAttempt = Time.time;
                }
            }
        }
        
        public async void ConnectToBridge()
        {
            if (isConnected) return;
            
            try
            {
                webSocket = new ClientWebSocket();
                Uri serverUri = new Uri(serverUrl);
                
                await webSocket.ConnectAsync(serverUri, cancellationTokenSource.Token);
                isConnected = true;
                
                Debug.Log($"Connected to IPC bridge at {serverUrl}");
                
                // Start listening for messages
                StartCoroutine(ListenForMessages());
                
            }
            catch (Exception ex)
            {
                Debug.LogError($"Failed to connect to IPC bridge: {ex.Message}");
                isConnected = false;
            }
        }
        
        private IEnumerator ListenForMessages()
        {
            byte[] buffer = new byte[4096];
            
            while (isConnected && webSocket.State == WebSocketState.Open)
            {
                try
                {
                    var result = webSocket.ReceiveAsync(new ArraySegment<byte>(buffer), cancellationTokenSource.Token);
                    
                    // Wait for result
                    while (!result.IsCompleted)
                    {
                        yield return null;
                    }
                    
                    if (result.Result.MessageType == WebSocketMessageType.Close)
                    {
                        await webSocket.CloseAsync(WebSocketCloseStatus.NormalClosure, "Connection closed", cancellationTokenSource.Token);
                        isConnected = false;
                        break;
                    }
                    
                    string message = Encoding.UTF8.GetString(buffer, 0, result.Result.Count);
                    HandleMessage(message);
                    
                }
                catch (Exception ex)
                {
                    Debug.LogError($"Error receiving message: {ex.Message}");
                    isConnected = false;
                    break;
                }
                
                yield return null;
            }
        }
        
        private void HandleMessage(string message)
        {
            Debug.Log($"Received message: {message}");
            
            OnCommandReceived?.Invoke(message);
            
            // Parse and handle message
            // Example: {"type": "spawn_mass", "position": [0, 0, 0]}
        }
        
        public async void SendStatus(string status)
        {
            if (!isConnected || webSocket.State != WebSocketState.Open)
            {
                return;
            }
            
            try
            {
                byte[] buffer = Encoding.UTF8.GetBytes(status);
                await webSocket.SendAsync(new ArraySegment<byte>(buffer), WebSocketMessageType.Text, true, cancellationTokenSource.Token);
                
                OnStatusSent?.Invoke(status);
            }
            catch (Exception ex)
            {
                Debug.LogError($"Error sending status: {ex.Message}");
            }
        }
        
        private void OnDestroy()
        {
            Disconnect();
        }
        
        public void Disconnect()
        {
            if (webSocket != null && isConnected)
            {
                webSocket.CloseAsync(WebSocketCloseStatus.NormalClosure, "Shutting down", CancellationToken.None);
                cancellationTokenSource?.Cancel();
            }
            
            isConnected = false;
        }
    }
}

