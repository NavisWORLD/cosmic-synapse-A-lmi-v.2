package world.navis.lighttoken;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Path;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import world.navis.lighttoken.model.BackendInfo;
import world.navis.lighttoken.model.ValidationResult;
import world.navis.lighttoken.nativebridge.JniNativeEngine;
import world.navis.lighttoken.nativebridge.NativeEngine;

class NativeEngineContractTest {
    @TempDir
    Path tempDir;

    @Test
    void loadsOnlyConfiguredNativeDirectoryAndReportsAbiOne() {
        Path nativeDir = Path.of(System.getProperty("lighttoken.native.dir"));
        try (NativeEngine engine = JniNativeEngine.load(nativeDir)) {
            assertEquals(1, engine.abiVersion());
            assertTrue(engine.contextHandle() != 0L);

            BackendInfo backend = engine.backendInfo();
            assertFalse(backend.activeBackend().isBlank());
        }
    }

    @Test
    void validatesSyntheticTokenThroughRealJni() {
        Path nativeDir = Path.of(System.getProperty("lighttoken.native.dir"));
        String token = """
                {
                  "token_id": "java-contract-token",
                  "timestamp": "2000-01-01T00:00:00+00:00",
                  "source_uri": "synthetic://java-contract",
                  "modality": "synthetic",
                  "raw_data_ref": "synthetic://java-contract/raw",
                  "metadata": {}
                }
                """;
        try (NativeEngine engine = JniNativeEngine.load(nativeDir)) {
            ValidationResult result = engine.validateJson(token);
            assertTrue(result.valid());
            assertEquals("java-contract-token", result.tokenId());
            assertEquals(0, result.embeddingDimension());
            assertEquals(0, result.spectralDimension());
        }
    }

    @Test
    void rejectsDirectoryWithoutApplicationNativeLibrary() {
        assertThrows(IllegalArgumentException.class, () -> JniNativeEngine.load(tempDir));
    }
}
