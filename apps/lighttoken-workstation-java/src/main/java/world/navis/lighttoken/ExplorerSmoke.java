package world.navis.lighttoken;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;
import javafx.application.Platform;
import world.navis.lighttoken.library.LibraryDatabase;
import world.navis.lighttoken.library.TokenRepository;
import world.navis.lighttoken.model.SimilarityMethod;
import world.navis.lighttoken.nativebridge.JniNativeEngine;
import world.navis.lighttoken.nativebridge.NativeEngine;
import world.navis.lighttoken.service.CollectionService;
import world.navis.lighttoken.service.SearchService;
import world.navis.lighttoken.service.ServiceExecutor;
import world.navis.lighttoken.ui.ExplorerViewModel;
import world.navis.lighttoken.ui.ResonanceExplorerView;

public final class ExplorerSmoke {
    private ExplorerSmoke() {}

    public static void main(String[] args) throws Exception {
        Path nativeDir = requiredPathProperty("lighttoken.native.dir");
        Path fixtureDir = requiredPathProperty("lighttoken.fixture.dir");
        Path temp = Files.createTempDirectory("lighttoken-javafx-smoke-");

        try (NativeEngine engine = JniNativeEngine.load(nativeDir);
                LibraryDatabase database = LibraryDatabase.open(temp.resolve("library.db"));
                ServiceExecutor workers = ServiceExecutor.create();
                CollectionService collections =
                        new CollectionService(engine, new TokenRepository(database), workers);
                SearchService searches =
                        new SearchService(engine, new TokenRepository(database), workers);
                ExplorerViewModel viewModel =
                        new ExplorerViewModel(engine, collections, searches)) {
            var snapshot = viewModel.openCollection(fixtureDir).join();
            String queryId = "00000000-0000-0000-0000-000000000004";
            viewModel.selectToken(queryId);
            var search = viewModel.search(queryId, SimilarityMethod.COSINE, 5).join();
            if (snapshot.tokens().size() != 5 || search.hits().isEmpty()) {
                throw new IllegalStateException("synthetic fixture/search smoke contract failed");
            }

            CountDownLatch latch = new CountDownLatch(1);
            AtomicReference<Throwable> failure = new AtomicReference<>();
            Platform.startup(
                    () -> {
                        try {
                            ResonanceExplorerView view = new ResonanceExplorerView(viewModel);
                            view.refreshFromViewModel();
                            if (view.embeddingPointCount() != 1536) {
                                throw new IllegalStateException(
                                        "expected 1536 embedding chart points");
                            }
                            if (view.spectrumPointCount() != 769) {
                                throw new IllegalStateException(
                                        "expected 769 spectrum chart points");
                            }
                            view.overlayCandidate(search.hits().getFirst().tokenId());
                        } catch (Throwable error) {
                            failure.set(error);
                        } finally {
                            latch.countDown();
                        }
                    });

            if (!latch.await(30, TimeUnit.SECONDS)) {
                throw new IllegalStateException("JavaFX smoke timed out");
            }
            Throwable error = failure.get();
            Platform.exit();
            if (error != null) {
                throw new RuntimeException("JavaFX smoke failed", error);
            }
            System.out.println(
                    "LightToken JavaFX smoke passed: tokens="
                            + snapshot.tokens().size()
                            + " embedding=1536 spectrum=769 backend="
                            + engine.backendInfo().activeBackend());
        } finally {
            deleteRecursively(temp);
        }
    }

    private static Path requiredPathProperty(String name) {
        String value = System.getProperty(name, "").trim();
        if (value.isEmpty()) {
            throw new IllegalArgumentException("missing system property: " + name);
        }
        Path path = Path.of(value).toAbsolutePath().normalize();
        if (!Files.exists(path)) {
            throw new IllegalArgumentException(name + " does not exist: " + path);
        }
        return path;
    }

    private static void deleteRecursively(Path root) throws Exception {
        if (!Files.exists(root)) {
            return;
        }
        try (var stream = Files.walk(root)) {
            for (Path path : stream.sorted(java.util.Comparator.reverseOrder()).toList()) {
                Files.deleteIfExists(path);
            }
        }
    }
}
