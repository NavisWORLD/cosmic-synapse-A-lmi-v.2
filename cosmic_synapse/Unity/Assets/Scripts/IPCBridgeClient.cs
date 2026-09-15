using UnityEngine;
using System;
using System.IO;
using System.Net.WebSockets;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace CosmicSynapse
{
    /// <summary>
    /// Unity client for the versioned Cosmic Synapse IPC WebSocket protocol.
    /// Network I/O uses Task-based async code; Unity lifecycle methods remain
    /// on the main thread and receive complete JSON messages as strings.
    /// </summary>
    public class IPCBridgeClient : MonoBehaviour
    {
        private const int protocolVersion = 1;

        [Header("IPC Settings")]
        [SerializeField] private string serverUrl = "ws://localhost:8765";
        [SerializeField] private bool autoConnect = true;
        [SerializeField] private float reconnectInterval = 5f;

        private ClientWebSocket webSocket;
        private CancellationTokenSource cancellationTokenSource;
        private Task receiveTask;
        private bool isConnected = false;
        private float lastReconnectAttempt = 0f;

        public Action<string> OnCommandReceived;
        public Action<string> OnStatusSent;

        [Serializable]
        private class StatusPayload
        {
            public string status;
        }

        [Serializable]
        private class StatusEnvelope
        {
            public int version = protocolVersion;
            public string type = "status";
            public StatusPayload payload;
        }

        [Serializable]
        private class MessageEnvelope
        {
            public int version;
            public string type;
        }

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
            if (!isConnected && autoConnect &&
                Time.time - lastReconnectAttempt > reconnectInterval)
            {
                ConnectToBridge();
                lastReconnectAttempt = Time.time;
            }
        }

        public async void ConnectToBridge()
        {
            if (isConnected) return;

            if (cancellationTokenSource == null || cancellationTokenSource.IsCancellationRequested)
            {
                cancellationTokenSource = new CancellationTokenSource();
            }

            try
            {
                if (webSocket != null)
                {
                    webSocket.Dispose();
                }

                webSocket = new ClientWebSocket();
                await webSocket.ConnectAsync(
                    new Uri(serverUrl),
                    cancellationTokenSource.Token
                );
                isConnected = true;
                Debug.Log($"Connected to IPC bridge at {serverUrl}");

                receiveTask = ListenForMessagesAsync(cancellationTokenSource.Token);
            }
            catch (Exception ex)
            {
                Debug.LogError($"Failed to connect to IPC bridge: {ex.Message}");
                isConnected = false;
            }
        }

        private async Task ListenForMessagesAsync(CancellationToken cancellationToken)
        {
            byte[] buffer = new byte[4096];

            try
            {
                while (!cancellationToken.IsCancellationRequested &&
                       webSocket != null &&
                       webSocket.State == WebSocketState.Open)
                {
                    using (var messageBuffer = new MemoryStream())
                    {
                        WebSocketReceiveResult result;
                        do
                        {
                            result = await webSocket.ReceiveAsync(
                                new ArraySegment<byte>(buffer),
                                cancellationToken
                            );

                            if (result.MessageType == WebSocketMessageType.Close)
                            {
                                await CloseSocketAsync("Remote endpoint closed", cancellationToken);
                                return;
                            }

                            messageBuffer.Write(buffer, 0, result.Count);
                        }
                        while (!result.EndOfMessage);

                        if (result.MessageType != WebSocketMessageType.Text)
                        {
                            continue;
                        }

                        string message = Encoding.UTF8.GetString(messageBuffer.ToArray());
                        HandleMessage(message);
                    }
                }
            }
            catch (OperationCanceledException)
            {
                // Normal shutdown path.
            }
            catch (Exception ex)
            {
                Debug.LogError($"Error receiving IPC message: {ex.Message}");
            }
            finally
            {
                isConnected = false;
            }
        }

        private void HandleMessage(string message)
        {
            try
            {
                MessageEnvelope envelope = JsonUtility.FromJson<MessageEnvelope>(message);
                if (envelope == null || envelope.version != protocolVersion)
                {
                    Debug.LogWarning("Ignoring IPC message with unsupported protocol version");
                    return;
                }

                if (string.Equals(envelope.type, "command", StringComparison.Ordinal))
                {
                    OnCommandReceived?.Invoke(message);
                }
            }
            catch (Exception ex)
            {
                Debug.LogError($"Invalid IPC JSON message: {ex.Message}");
            }
        }

        public async void SendStatus(string status)
        {
            if (!isConnected || webSocket == null || webSocket.State != WebSocketState.Open)
            {
                return;
            }

            try
            {
                var envelope = new StatusEnvelope
                {
                    payload = new StatusPayload { status = status }
                };
                string json = JsonUtility.ToJson(envelope);
                byte[] bytes = Encoding.UTF8.GetBytes(json);
                await webSocket.SendAsync(
                    new ArraySegment<byte>(bytes),
                    WebSocketMessageType.Text,
                    true,
                    cancellationTokenSource.Token
                );
                OnStatusSent?.Invoke(status);
            }
            catch (Exception ex)
            {
                Debug.LogError($"Error sending IPC status: {ex.Message}");
            }
        }

        private async Task CloseSocketAsync(string reason, CancellationToken cancellationToken)
        {
            if (webSocket == null) return;

            if (webSocket.State == WebSocketState.Open ||
                webSocket.State == WebSocketState.CloseReceived)
            {
                await webSocket.CloseAsync(
                    WebSocketCloseStatus.NormalClosure,
                    reason,
                    cancellationToken
                );
            }
            isConnected = false;
        }

        private void OnDestroy()
        {
            Disconnect();
        }

        public void Disconnect()
        {
            isConnected = false;
            cancellationTokenSource?.Cancel();

            if (webSocket != null)
            {
                webSocket.Dispose();
                webSocket = null;
            }
        }
    }
}
