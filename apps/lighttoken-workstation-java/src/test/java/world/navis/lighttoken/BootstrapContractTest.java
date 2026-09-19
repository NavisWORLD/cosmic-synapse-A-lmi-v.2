package world.navis.lighttoken;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;

import java.lang.reflect.Method;
import java.nio.file.Files;
import java.nio.file.Path;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.ResultSet;
import java.sql.Statement;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class BootstrapContractTest {
    @TempDir
    Path tempDir;

    @Test
    void sqliteDependencyPersistsRowsAcrossReopen() throws Exception {
        Path database = tempDir.resolve("bootstrap.db");
        String url = "jdbc:sqlite:" + database;
        try (Connection connection = DriverManager.getConnection(url);
             Statement statement = connection.createStatement()) {
            statement.executeUpdate("CREATE TABLE sample(value TEXT NOT NULL)");
            statement.executeUpdate("INSERT INTO sample(value) VALUES ('persisted')");
        }

        try (Connection connection = DriverManager.getConnection(url);
             Statement statement = connection.createStatement();
             ResultSet result = statement.executeQuery("SELECT value FROM sample")) {
            assertEquals("persisted", result.getString(1));
        }
    }

    @Test
    void nativeEngineContractExistsAndReportsAbiOne() throws Exception {
        String configured = System.getProperty("lighttoken.native.dir", "");
        assertFalse(configured.isBlank(), "CI must supply the application-owned native directory");
        Path nativeDir = Path.of(configured);
        assertFalse(Files.notExists(nativeDir), "native directory must exist");

        Class<?> engineType =
                Class.forName("world.navis.lighttoken.nativebridge.JniNativeEngine");
        Method load = engineType.getMethod("load", Path.class);
        Object engine = load.invoke(null, nativeDir);
        try {
            Method abiVersion = engineType.getMethod("abiVersion");
            assertEquals(1, ((Number) abiVersion.invoke(engine)).intValue());
        } finally {
            engineType.getMethod("close").invoke(engine);
        }
    }
}
