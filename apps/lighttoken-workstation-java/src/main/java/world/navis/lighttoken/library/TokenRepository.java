package world.navis.lighttoken.library;

import java.nio.file.Path;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Statement;
import java.util.ArrayList;
import java.util.List;
import world.navis.lighttoken.model.QueryRecord;
import world.navis.lighttoken.model.TokenSummary;

public final class TokenRepository {
    private final LibraryDatabase database;

    public TokenRepository(LibraryDatabase database) {
        this.database = database;
    }

    public long upsertSource(
            Path sourcePath, String sourceKind, String contentSha256, String indexedAt)
            throws SQLException {
        String path = sourcePath.toAbsolutePath().normalize().toString();
        try (PreparedStatement statement =
                database.connection()
                        .prepareStatement(
                                """
                                INSERT INTO sources(source_path, source_kind, content_sha256, indexed_at)
                                VALUES (?, ?, ?, ?)
                                ON CONFLICT(source_path) DO UPDATE SET
                                  source_kind=excluded.source_kind,
                                  content_sha256=excluded.content_sha256,
                                  indexed_at=excluded.indexed_at
                                """)) {
            statement.setString(1, path);
            statement.setString(2, sourceKind);
            statement.setString(3, contentSha256);
            statement.setString(4, indexedAt);
            statement.executeUpdate();
        }
        try (PreparedStatement statement =
                database.connection()
                        .prepareStatement("SELECT source_id FROM sources WHERE source_path = ?")) {
            statement.setString(1, path);
            try (ResultSet result = statement.executeQuery()) {
                if (!result.next()) {
                    throw new SQLException("source row was not persisted");
                }
                return result.getLong(1);
            }
        }
    }

    public void upsertToken(TokenSummary token, long sourceId, int cacheRevision)
            throws SQLException {
        try (PreparedStatement statement =
                database.connection()
                        .prepareStatement(
                                """
                                INSERT INTO tokens(
                                  token_id, source_id, source_uri, modality, raw_data_ref, timestamp, cache_revision
                                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                                ON CONFLICT(token_id) DO UPDATE SET
                                  source_id=excluded.source_id,
                                  source_uri=excluded.source_uri,
                                  modality=excluded.modality,
                                  raw_data_ref=excluded.raw_data_ref,
                                  timestamp=excluded.timestamp,
                                  cache_revision=excluded.cache_revision
                                """)) {
            statement.setString(1, token.tokenId());
            statement.setLong(2, sourceId);
            statement.setString(3, token.sourceUri());
            statement.setString(4, token.modality());
            statement.setString(5, token.rawDataRef());
            statement.setString(6, token.timestamp());
            statement.setInt(7, cacheRevision);
            statement.executeUpdate();
        }
    }

    public List<TokenSummary> listTokens() throws SQLException {
        List<TokenSummary> tokens = new ArrayList<>();
        try (PreparedStatement statement =
                        database.connection()
                                .prepareStatement(
                                        """
                                        SELECT token_id, timestamp, source_uri, modality, raw_data_ref
                                        FROM tokens ORDER BY token_id ASC
                                        """);
                ResultSet result = statement.executeQuery()) {
            while (result.next()) {
                tokens.add(
                        new TokenSummary(
                                result.getString("token_id"),
                                result.getString("timestamp"),
                                result.getString("source_uri"),
                                result.getString("modality"),
                                result.getString("raw_data_ref")));
            }
        }
        return List.copyOf(tokens);
    }

    public long recordQuery(
            String queryTokenId, String method, String backend, String createdAt)
            throws SQLException {
        try (PreparedStatement statement =
                database.connection()
                        .prepareStatement(
                                "INSERT INTO queries(query_token_id, method, backend, created_at) VALUES (?, ?, ?, ?)",
                                Statement.RETURN_GENERATED_KEYS)) {
            statement.setString(1, queryTokenId);
            statement.setString(2, method);
            statement.setString(3, backend);
            statement.setString(4, createdAt);
            statement.executeUpdate();
            try (ResultSet keys = statement.getGeneratedKeys()) {
                if (!keys.next()) {
                    throw new SQLException("query id was not generated");
                }
                return keys.getLong(1);
            }
        }
    }

    public void recordQueryHit(long queryId, int rank, String tokenId, float score)
            throws SQLException {
        try (PreparedStatement statement =
                database.connection()
                        .prepareStatement(
                                "INSERT INTO query_hits(query_id, rank, token_id, score) VALUES (?, ?, ?, ?)")) {
            statement.setLong(1, queryId);
            statement.setInt(2, rank);
            statement.setString(3, tokenId);
            statement.setFloat(4, score);
            statement.executeUpdate();
        }
    }

    public List<QueryRecord> listQueries() throws SQLException {
        List<QueryRecord> queries = new ArrayList<>();
        try (PreparedStatement statement =
                        database.connection()
                                .prepareStatement(
                                        """
                                        SELECT q.query_id, q.query_token_id, q.method, q.backend, q.created_at,
                                               COUNT(h.rank) AS hit_count
                                        FROM queries q
                                        LEFT JOIN query_hits h ON h.query_id = q.query_id
                                        GROUP BY q.query_id, q.query_token_id, q.method, q.backend, q.created_at
                                        ORDER BY q.query_id ASC
                                        """);
                ResultSet result = statement.executeQuery()) {
            while (result.next()) {
                queries.add(
                        new QueryRecord(
                                result.getLong("query_id"),
                                result.getString("query_token_id"),
                                result.getString("method"),
                                result.getString("backend"),
                                result.getString("created_at"),
                                result.getInt("hit_count")));
            }
        }
        return List.copyOf(queries);
    }

    public void rebuildCache() throws SQLException {
        boolean originalAutoCommit = database.connection().getAutoCommit();
        database.connection().setAutoCommit(false);
        try (Statement statement = database.connection().createStatement()) {
            statement.executeUpdate("DELETE FROM query_hits");
            statement.executeUpdate("DELETE FROM queries");
            statement.executeUpdate("DELETE FROM tokens");
            statement.executeUpdate("DELETE FROM sources");
            database.connection().commit();
        } catch (SQLException error) {
            database.connection().rollback();
            throw error;
        } finally {
            database.connection().setAutoCommit(originalAutoCommit);
        }
    }
}
