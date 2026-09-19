package world.navis.lighttoken.library;

import java.sql.Connection;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Statement;

public final class LibraryMigrations {
    public static final int SCHEMA_VERSION = 1;

    private LibraryMigrations() {}

    public static void migrate(Connection connection) throws SQLException {
        boolean originalAutoCommit = connection.getAutoCommit();
        connection.setAutoCommit(false);
        try (Statement statement = connection.createStatement()) {
            statement.executeUpdate(
                    "CREATE TABLE IF NOT EXISTS schema_version(version INTEGER NOT NULL)");
            int current = readVersion(statement);
            if (current == 0) {
                createSchemaOne(statement);
                statement.executeUpdate("INSERT INTO schema_version(version) VALUES (1)");
            } else if (current != SCHEMA_VERSION) {
                throw new SQLException(
                        "unsupported LightToken library schema version " + current);
            }
            connection.commit();
        } catch (SQLException error) {
            connection.rollback();
            throw error;
        } finally {
            connection.setAutoCommit(originalAutoCommit);
        }
    }

    private static int readVersion(Statement statement) throws SQLException {
        try (ResultSet result =
                statement.executeQuery("SELECT version FROM schema_version LIMIT 1")) {
            return result.next() ? result.getInt(1) : 0;
        }
    }

    private static void createSchemaOne(Statement statement) throws SQLException {
        statement.executeUpdate(
                """
                CREATE TABLE sources(
                  source_id INTEGER PRIMARY KEY,
                  source_path TEXT NOT NULL UNIQUE,
                  source_kind TEXT NOT NULL,
                  content_sha256 TEXT,
                  indexed_at TEXT NOT NULL
                )
                """);
        statement.executeUpdate(
                """
                CREATE TABLE tokens(
                  token_id TEXT PRIMARY KEY,
                  source_id INTEGER NOT NULL REFERENCES sources(source_id) ON DELETE CASCADE,
                  source_uri TEXT NOT NULL,
                  modality TEXT NOT NULL,
                  raw_data_ref TEXT NOT NULL,
                  timestamp TEXT NOT NULL,
                  cache_revision INTEGER NOT NULL
                )
                """);
        statement.executeUpdate(
                """
                CREATE TABLE queries(
                  query_id INTEGER PRIMARY KEY,
                  query_token_id TEXT NOT NULL,
                  method TEXT NOT NULL,
                  backend TEXT NOT NULL,
                  created_at TEXT NOT NULL
                )
                """);
        statement.executeUpdate(
                """
                CREATE TABLE query_hits(
                  query_id INTEGER NOT NULL REFERENCES queries(query_id) ON DELETE CASCADE,
                  rank INTEGER NOT NULL,
                  token_id TEXT NOT NULL,
                  score REAL NOT NULL,
                  PRIMARY KEY(query_id, rank)
                )
                """);
    }
}
