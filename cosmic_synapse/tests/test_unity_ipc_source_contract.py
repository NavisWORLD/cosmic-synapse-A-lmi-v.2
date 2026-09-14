from pathlib import Path


UNITY_CLIENT = Path("cosmic_synapse/Unity/Assets/Scripts/IPCBridgeClient.cs")


def test_unity_receive_loop_uses_task_based_async_not_coroutine_await_mix():
    source = UNITY_CLIENT.read_text(encoding="utf-8")
    assert "async Task ListenForMessagesAsync" in source
    assert "IEnumerator ListenForMessages" not in source
    assert "StartCoroutine(ListenForMessages" not in source


def test_unity_client_emits_versioned_status_envelope():
    source = UNITY_CLIENT.read_text(encoding="utf-8")
    assert "protocolVersion = 1" in source
    assert "type = \"status\"" in source
