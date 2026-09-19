package world.navis.lighttoken;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Path;
import java.util.List;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import world.navis.lighttoken.library.LibraryDatabase;
import world.navis.lighttoken.library.TokenRepository;
import world.navis.lighttoken.model.QueryRecord;
import world.navis.lighttoken.model.TokenSummary;

class LibraryDatabaseContractTest {
    @TempDir
    Path tempDir;

    @Test
    void schemaOnePersistsSourcesTokensQueriesAndHitsAcrossReopen() throws Exception {
        Path databasePath = tempDir.resolve("library.db");
        TokenSummary token = new TokenSummary(
                "token-a",
                "2000-01-01T00:00:00+00:00",
                "synthetic://collection/a",
                "synthetic",
                "synthetic://collection/a/raw");

        try (LibraryDatabase database = LibraryDatabase.open(databasePath)) {
            assertEquals(1, database.schemaVersion());
            TokenRepository repository = new TokenRepository(database);
            long sourceId = repository.upsertSource(
                    tempDir.resolve("collection.jsonl"), "jsonl", "abc123", "2000-01-01T00:00:00+00:00");
            repository.upsertToken(token, sourceId, 1);
            long queryId = repository.recordQuery(
                    token.tokenId(), "cosine", "rust", "2000-01-01T00:00:01+00:00");
            repository.recordQueryHit(queryId, 1, token.tokenId(), 1.0f);
        }

        try (LibraryDatabase reopened = LibraryDatabase.open(databasePath)) {
            TokenRepository repository = new TokenRepository(reopened);
            assertEquals(List.of(token), repository.listTokens());
            List<QueryRecord> queries = repository.listQueries();
            assertEquals(1, queries.size());
            assertEquals("token-a", queries.getFirst().queryTokenId());
            assertEquals("cosine", queries.getFirst().method());
            assertEquals("rust", queries.getFirst().backend());
            assertEquals(1, queries.getFirst().hitCount());

            repository.rebuildCache();
            assertTrue(repository.listTokens().isEmpty());
            assertTrue(repository.listQueries().isEmpty());
        }
    }
}
