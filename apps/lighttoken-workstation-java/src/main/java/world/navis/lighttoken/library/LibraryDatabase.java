package world.navis.lighttoken.library;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Statement;

public final class LibraryDatabase implements AutoCloseable {
    private final Path path;
    private final Connection connection;

    private LibraryDatabase(Path path, Connection connection) {
        this.path = path;
        this.connection = connection;
    }

    public static LibraryDatabase open(Path databasePath) throws SQLException, IOException {
        Path path = databasePath.toAbsolutePath().normalize();
        Path parent = path.getParent();
        if (parent != null) {
            Files.createDirectories(parent);
        }
        Connection connection = DriverManager.getConnection("jdbc:sqlite:" + path);
        try {
            try (Statement statement = connection.createStatement()) {
                statement.execute("PRAGMA foreign_keys = ON");
            }
            LibraryMigrations.migrate(connection);
            return new LibraryDatabase(path, connection);
        } catch (SQLException error) {
            connection.close();
            throw error;
        }
    }

    Connection connection() {
        return connection;
    }

    public Path path() {
        return path;
    }

    public int schemaVersion() throws SQLException {
        try (Statement statement = connection.createStatement();
                ResultSet result =
                        statement.executeQuery("SELECT version FROM schema_version LIMIT 1")) {
            if (!result.next()) {
                throw new SQLException("schema_version is empty");
            }
            return result.getInt(1);
        }
    }

    @Override
    public void close() throws SQLException {
        connection.close();
    }
}
