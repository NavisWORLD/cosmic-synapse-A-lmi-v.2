package world.navis.lighttoken.ui;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;

import java.nio.file.Path;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import world.navis.lighttoken.library.LibraryDatabase;
import world.navis.lighttoken.library.TokenRepository;
import world.navis.lighttoken.model.SimilarityMethod;
import world.navis.lighttoken.nativebridge.JniNativeEngine;
import world.navis.lighttoken.nativebridge.NativeEngine;
import world.navis.lighttoken.service.CollectionService;
import world.navis.lighttoken.service.SearchService;
import world.navis.lighttoken.service.ServiceExecutor;

class ExplorerViewModelTest {
    @TempDir
    Path tempDir;

    @Test
    void viewModelUsesNativeServicesForOpenSelectAndSearch() throws Exception {
        Path fixtureDir = Path.of(System.getProperty("lighttoken.fixture.dir"));
        Path nativeDir = Path.of(System.getProperty("lighttoken.native.dir"));
        try (NativeEngine engine = JniNativeEngine.load(nativeDir);
                LibraryDatabase database = LibraryDatabase.open(tempDir.resolve("ui.db"));
                ServiceExecutor workers = ServiceExecutor.create();
                CollectionService collections =
                        new CollectionService(engine, new TokenRepository(database), workers);
                SearchService searches =
                        new SearchService(engine, new TokenRepository(database), workers);
                ExplorerViewModel viewModel =
                        new ExplorerViewModel(engine, collections, searches)) {
            var snapshot = viewModel.openCollection(fixtureDir).join();
            assertSame(snapshot, viewModel.snapshot());

            String queryId = "00000000-0000-0000-0000-000000000004";
            var selected = viewModel.selectToken(queryId);
            assertEquals(1536, selected.vectors().embedding().length);
            assertEquals(769, selected.vectors().spectralPower().length);

            var result = viewModel.search(queryId, SimilarityMethod.COSINE, 5).join();
            assertEquals(queryId, result.hits().getFirst().tokenId());
            assertEquals(engine.backendInfo().activeBackend(), viewModel.backendInfo().activeBackend());
        }
    }
}
