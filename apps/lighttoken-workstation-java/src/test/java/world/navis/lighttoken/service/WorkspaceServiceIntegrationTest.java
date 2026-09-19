package world.navis.lighttoken.service;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.util.HexFormat;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.concurrent.CompletionException;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import world.navis.lighttoken.library.LibraryDatabase;
import world.navis.lighttoken.library.TokenRepository;
import world.navis.lighttoken.nativebridge.JniNativeEngine;
import world.navis.lighttoken.nativebridge.NativeEngine;

class WorkspaceServiceIntegrationTest {
    @TempDir
    Path tempDir;

    private static Path required(String property) {
        return Path.of(System.getProperty(property)).toAbsolutePath().normalize();
    }

    private static Map<String, String> hashes(Path root) throws Exception {
        Map<String, String> result = new LinkedHashMap<>();
        try (var stream = Files.walk(root)) {
            for (Path path : stream.filter(Files::isRegularFile).sorted().toList()) {
                byte[] digest = MessageDigest.getInstance("SHA-256").digest(Files.readAllBytes(path));
                result.put(root.relativize(path).toString(), HexFormat.of().formatHex(digest));
            }
        }
        return result;
    }

    @Test
    void verifiedWorkspaceAndCosmosOpenReadOnlyThroughRustJni() throws Exception {
        Path workspace = required("lighttoken.almi.workspace");
        Path bundle = required("lighttoken.almi.bundle");
        Path corrupt = required("lighttoken.almi.corrupt_bundle");
        Map<String, String> workspaceBefore = hashes(workspace);
        String bundleBefore = HexFormat.of().formatHex(
                MessageDigest.getInstance("SHA-256").digest(Files.readAllBytes(bundle)));

        try (NativeEngine engine = JniNativeEngine.load(required("lighttoken.native.dir"));
                LibraryDatabase database = LibraryDatabase.open(tempDir.resolve("workspace.db"));
                ServiceExecutor workers = ServiceExecutor.create();
                WorkspaceService service =
                        new WorkspaceService(engine, new TokenRepository(database), workers)) {
            var directNative = engine.readWorkspace(workspace);
            assertFalse(directNative.bundleVerified());
            assertEquals(1, directNative.tokens().size());
            assertTrue(directNative.tokens().getFirst().verified());
            assertTrue(directNative.tokens().getFirst().rawResolvable());

            var bundleNative = engine.readCosmos(bundle);
            assertTrue(bundleNative.bundleVerified());
            assertEquals(1, bundleNative.tokens().size());

            var direct = service.openWorkspace(workspace).join();
            assertEquals(1, direct.tokens().size());
            assertTrue(direct.tokens().getFirst().verification().verified());

            var imported = service.openCosmos(bundle).join();
            assertEquals(1, imported.tokens().size());
            assertTrue(imported.tokens().getFirst().verification().verified());

            CompletionException failure =
                    assertThrows(CompletionException.class, () -> service.openCosmos(corrupt).join());
            assertTrue(failure.getCause().getMessage().toLowerCase().contains("integrity"));
        }

        assertEquals(workspaceBefore, hashes(workspace));
        assertEquals(
                bundleBefore,
                HexFormat.of().formatHex(
                        MessageDigest.getInstance("SHA-256").digest(Files.readAllBytes(bundle))));
    }
}
