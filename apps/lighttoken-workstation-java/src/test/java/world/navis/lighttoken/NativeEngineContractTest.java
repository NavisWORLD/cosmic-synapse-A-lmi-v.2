package world.navis.lighttoken;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Path;
import java.util.List;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import world.navis.lighttoken.model.BackendInfo;
import world.navis.lighttoken.model.ComparisonResult;
import world.navis.lighttoken.model.SearchRequest;
import world.navis.lighttoken.model.SearchResult;
import world.navis.lighttoken.model.SimilarityMethod;
import world.navis.lighttoken.model.TokenVectors;
import world.navis.lighttoken.model.ValidationResult;
import world.navis.lighttoken.nativebridge.JniNativeEngine;
import world.navis.lighttoken.nativebridge.NativeEngine;

class NativeEngineContractTest {
    @TempDir
    Path tempDir;

    private static Path nativeDir() {
        return Path.of(System.getProperty("lighttoken.native.dir"));
    }

    private static String tokenJson(String tokenId, float sign) {
        StringBuilder embedding = new StringBuilder();
        for (int index = 0; index < 1536; index++) {
            if (index > 0) embedding.append(',');
            embedding.append(sign * ((index % 17) + 1) / 17.0f);
        }
        StringBuilder real = new StringBuilder();
        StringBuilder imag = new StringBuilder();
        for (int index = 0; index < 769; index++) {
            if (index > 0) {
                real.append(',');
                imag.append(',');
            }
            real.append(sign * ((index % 13) + 1));
            imag.append(sign * ((index % 7) + 1) / 7.0f);
        }
        return """
                {
                  "token_id": "%s",
                  "timestamp": "2000-01-01T00:00:00+00:00",
                  "source_uri": "synthetic://java-contract/%s",
                  "modality": "synthetic",
                  "raw_data_ref": "synthetic://java-contract/%s/raw",
                  "metadata": {},
                  "joint_embedding": [%s],
                  "spectral_signature_real": [%s],
                  "spectral_signature_imag": [%s]
                }
                """.formatted(tokenId, tokenId, tokenId, embedding, real, imag);
    }

    @Test
    void loadsOnlyConfiguredNativeDirectoryAndReportsAbiOne() {
        try (NativeEngine engine = JniNativeEngine.load(nativeDir())) {
            assertEquals(1, engine.abiVersion());
            assertTrue(engine.contextHandle() != 0L);

            BackendInfo backend = engine.backendInfo();
            assertFalse(backend.activeBackend().isBlank());
        }
    }

    @Test
    void validatesComparesSearchesAndReturnsChartVectorsThroughRealJni() {
        String left = tokenJson("java-left", 1.0f);
        String right = tokenJson("java-right", -1.0f);
        try (NativeEngine engine = JniNativeEngine.load(nativeDir())) {
            ValidationResult validation = engine.validateJson(left);
            assertTrue(validation.valid());
            assertEquals("java-left", validation.tokenId());
            assertEquals(1536, validation.embeddingDimension());
            assertEquals(769, validation.spectralDimension());

            ComparisonResult comparison =
                    engine.compareJson(left, left, SimilarityMethod.COSINE);
            assertEquals("java-left", comparison.leftTokenId());
            assertEquals(1.0f, comparison.score(), 1.0e-5f);

            SearchResult search = engine.searchJson(
                    left,
                    "[" + left + "," + right + "]",
                    new SearchRequest(SimilarityMethod.COSINE, 2, null, null, null));
            assertEquals("java-left", search.queryTokenId());
            assertEquals(List.of("java-left", "java-right"),
                    search.hits().stream().map(hit -> hit.tokenId()).toList());

            TokenVectors vectors = engine.vectorsJson(left);
            assertEquals("java-left", vectors.tokenId());
            assertEquals(1536, vectors.embedding().length);
            assertEquals(769, vectors.spectralPower().length);
        }
    }

    @Test
    void rejectsDirectoryWithoutApplicationNativeLibrary() {
        assertThrows(IllegalArgumentException.class, () -> JniNativeEngine.load(tempDir));
    }
}
