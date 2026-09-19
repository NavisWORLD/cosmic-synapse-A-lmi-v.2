package world.navis.lighttoken.ui;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;

import org.junit.jupiter.api.Test;

class UiSemanticContractTest {
    @Test
    void chartLanguageStaysInEmbeddingDomain() {
        assertEquals("Embedding spectrum", ResonanceExplorerView.EMBEDDING_CHART_TITLE);
        assertEquals("Spectral bin", ResonanceExplorerView.SPECTRAL_X_AXIS_LABEL);

        String labels =
                String.join(
                                " ",
                                ResonanceExplorerView.EMBEDDING_CHART_TITLE,
                                ResonanceExplorerView.EMBEDDING_X_AXIS_LABEL,
                                ResonanceExplorerView.SPECTRAL_CHART_TITLE,
                                ResonanceExplorerView.SPECTRAL_X_AXIS_LABEL)
                        .toLowerCase();
        assertFalse(labels.contains("hz"));
        assertFalse(labels.contains("frequency"));
    }
}
