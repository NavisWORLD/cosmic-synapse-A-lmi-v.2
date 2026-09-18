package world.navis.lighttoken.service;

import static org.junit.jupiter.api.Assertions.assertThrows;

import java.nio.file.Path;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class ExportBoundaryTest {
    @TempDir
    Path tempDir;

    @Test
    void exportsRequireExplicitFileDestinations() {
        try (ServiceExecutor workers = ServiceExecutor.create();
                ExportService exports = new ExportService(workers)) {
            assertThrows(IllegalArgumentException.class, () -> exports.requireExplicitFile(tempDir));
            assertThrows(IllegalArgumentException.class, () -> exports.requireExplicitFile(Path.of("")));
        }
    }
}
