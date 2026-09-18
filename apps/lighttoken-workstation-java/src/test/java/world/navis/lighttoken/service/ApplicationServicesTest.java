package world.navis.lighttoken.service;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.util.HexFormat;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import world.navis.lighttoken.library.LibraryDatabase;
import world.navis.lighttoken.library.TokenRepository;
import world.navis.lighttoken.model.CollectionSnapshot;
import world.navis.lighttoken.model.QueryResult;
import world.navis.lighttoken.model.SearchFilters;
import world.navis.lighttoken.model.SimilarityMethod;
import world.navis.lighttoken.model.TokenDetail;
import world.navis.lighttoken.nativebridge.JniNativeEngine;
import world.navis.lighttoken.nativebridge.NativeEngine;

class ApplicationServicesTest {
    private static final ObjectMapper JSON = new ObjectMapper();

    @TempDir
    Path tempDir;

    private static Path fixtureDir() {
        return Path.of(System.getProperty("lighttoken.fixture.dir")).toAbsolutePath().normalize();
    }

    private static Path nativeDir() {
        return Path.of(System.getProperty("lighttoken.native.dir")).toAbsolutePath().normalize();
    }

    private static Map<String, String> hashes(Path directory) throws Exception {
        Map<String, String> hashes = new LinkedHashMap<>();
        try (var stream = Files.list(directory)) {
            for (Path path : stream.filter(Files::isRegularFile).sorted().toList()) {
                byte[] digest = MessageDigest.getInstance("SHA-256").digest(Files.readAllBytes(path));
                hashes.put(path.getFileName().toString(), HexFormat.of().formatHex(digest));
            }
        }
        return hashes;
    }

    private static double oracleScore(String left, String right, String method) throws Exception {
        JsonNode oracle = JSON.readTree(fixtureDir().resolve("similarity.json").toFile());
        for (JsonNode pair : oracle.path("pairs")) {
            boolean direct = pair.path("a").asText().equals(left) && pair.path("b").asText().equals(right);
            boolean reverse = pair.path("a").asText().equals(right) && pair.path("b").asText().equals(left);
            if ((direct || reverse) && pair.path("method").asText().equals(method)) {
                return pair.path("score").asDouble();
            }
        }
        throw new IllegalStateException("oracle pair not found");
    }

    @Test
    void workerCountIsBoundedByContract() {
        assertEquals(1, ServiceExecutor.workerCountForProcessors(1));
        assertEquals(1, ServiceExecutor.workerCountForProcessors(2));
        assertEquals(2, ServiceExecutor.workerCountForProcessors(4));
        assertEquals(4, ServiceExecutor.workerCountForProcessors(8));
        assertEquals(4, ServiceExecutor.workerCountForProcessors(64));
    }

    @Test
    void realJniCollectionSearchAndExportsPreserveGoldenSources() throws Exception {
        Map<String, String> before = hashes(fixtureDir());
        Path databasePath = tempDir.resolve("library.db");

        try (NativeEngine engine = JniNativeEngine.load(nativeDir());
                LibraryDatabase database = LibraryDatabase.open(databasePath);
                ServiceExecutor workers = ServiceExecutor.create();
                CollectionService collections =
                        new CollectionService(engine, new TokenRepository(database), workers);
                SearchService searches =
                        new SearchService(engine, new TokenRepository(database), workers);
                ExportService exports = new ExportService(workers);
                WorkspaceService workspaces = new WorkspaceService(collections)) {

            CompletableFuture<CollectionSnapshot> openedFuture = workspaces.openWorkspace(fixtureDir());
            CollectionSnapshot snapshot = openedFuture.join();
            assertEquals(5, snapshot.tokens().size());
            assertTrue(snapshot.tokens().stream().allMatch(token -> token.vectors().embedding().length == 1536));
            assertTrue(snapshot.tokens().stream().allMatch(token -> token.vectors().spectralPower().length == 769));

            String queryId = "00000000-0000-0000-0000-000000000004";
            String randomId = "00000000-0000-0000-0000-000000000005";

            QueryResult cosine = searches
                    .search(snapshot, queryId, SimilarityMethod.COSINE, 5, null, SearchFilters.none())
                    .join();
            assertEquals(queryId, cosine.hits().getFirst().tokenId());
            assertEquals(1.0f, cosine.hits().getFirst().score(), 1.0e-5f);
            double expectedRandomCosine = oracleScore("active_sinusoid", "active_random", "cosine");
            double actualRandomCosine = cosine.hits().stream()
                    .filter(hit -> hit.tokenId().equals(randomId))
                    .findFirst()
                    .orElseThrow()
                    .score();
            assertEquals(expectedRandomCosine, actualRandomCosine, 2.0e-4);

            QueryResult correlation = searches
                    .search(
                            snapshot,
                            queryId,
                            SimilarityMethod.POWER_CORRELATION,
                            5,
                            null,
                            SearchFilters.none())
                    .join();
            assertEquals(queryId, correlation.hits().getFirst().tokenId());
            assertEquals("power_correlation", correlation.method().wireName());
            assertFalse(new TokenRepository(database).listQueries().isEmpty());

            TokenDetail query = snapshot.token(queryId);
            Path tokenExport = tempDir.resolve("exports/query-token.json");
            Path resultExport = tempDir.resolve("exports/query-result.json");
            exports.exportToken(query, tokenExport).join();
            exports.exportQueryResults(cosine, resultExport).join();

            JsonNode exportedToken = JSON.readTree(tokenExport.toFile());
            assertEquals(queryId, exportedToken.path("token_id").asText());
            JsonNode exportedResult = JSON.readTree(resultExport.toFile());
            assertEquals(queryId, exportedResult.path("query_token_id").asText());
            assertEquals("cosine", exportedResult.path("method").asText());
            assertFalse(exportedResult.path("backend").asText().isBlank());
            assertTrue(exportedResult.path("hits").isArray());
            assertTrue(exportedResult.path("hits").size() >= 2);
        }

        assertEquals(before, hashes(fixtureDir()), "open/search/export must not mutate source fixtures");
    }
}
