package world.navis.lighttoken.service;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HexFormat;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.CompletableFuture;
import world.navis.lighttoken.library.TokenRepository;
import world.navis.lighttoken.model.CollectionSnapshot;
import world.navis.lighttoken.model.TokenDetail;
import world.navis.lighttoken.model.TokenSummary;
import world.navis.lighttoken.nativebridge.NativeEngine;

public final class CollectionService implements AutoCloseable {
    private static final ObjectMapper JSON = new ObjectMapper();

    private final NativeEngine engine;
    private final TokenRepository repository;
    private final ServiceExecutor workers;

    public CollectionService(
            NativeEngine engine, TokenRepository repository, ServiceExecutor workers) {
        this.engine = Objects.requireNonNull(engine, "engine");
        this.repository = Objects.requireNonNull(repository, "repository");
        this.workers = Objects.requireNonNull(workers, "workers");
    }

    public CompletableFuture<CollectionSnapshot> open(Path source) {
        Path normalized = Objects.requireNonNull(source, "source").toAbsolutePath().normalize();
        return workers.submit(() -> openBlocking(normalized));
    }

    private CollectionSnapshot openBlocking(Path source) throws Exception {
        if (!Files.exists(source)) {
            throw new IllegalArgumentException("collection source does not exist: " + source);
        }
        List<Path> tokenPaths;
        if (Files.isDirectory(source)) {
            try (var stream = Files.list(source)) {
                tokenPaths =
                        stream.filter(Files::isRegularFile)
                                .filter(
                                        path -> {
                                            String name = path.getFileName().toString();
                                            return name.startsWith("active_") && name.endsWith(".json");
                                        })
                                .sorted(Comparator.comparing(path -> path.getFileName().toString()))
                                .toList();
            }
        } else {
            tokenPaths = List.of(source);
        }
        if (tokenPaths.isEmpty()) {
            throw new IllegalArgumentException("collection contains no active LightToken JSON files");
        }

        List<TokenDetail> details = new ArrayList<>();
        List<String> jsonValues = new ArrayList<>();
        for (Path path : tokenPaths) {
            byte[] bytes = Files.readAllBytes(path);
            String json = new String(bytes, StandardCharsets.UTF_8).trim();
            var validation = engine.validateJson(json);
            if (!validation.valid()) {
                throw new IllegalArgumentException("native validation rejected " + path);
            }
            JsonNode node = JSON.readTree(json);
            TokenSummary summary =
                    new TokenSummary(
                            validation.tokenId(),
                            requiredText(node, "timestamp"),
                            requiredText(node, "source_uri"),
                            requiredText(node, "modality"),
                            requiredText(node, "raw_data_ref"));
            long sourceId =
                    repository.upsertSource(
                            path,
                            "lighttoken-json",
                            sha256(bytes),
                            summary.timestamp());
            repository.upsertToken(summary, sourceId, 1);
            details.add(new TokenDetail(summary, json + "\n", engine.vectorsJson(json), path));
            jsonValues.add(json);
        }
        return new CollectionSnapshot(source, details, "[" + String.join(",", jsonValues) + "]");
    }

    private static String requiredText(JsonNode node, String field) {
        String value = node.path(field).asText("");
        if (value.isBlank()) {
            throw new IllegalArgumentException("LightToken is missing " + field);
        }
        return value;
    }

    private static String sha256(byte[] bytes) throws Exception {
        return HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(bytes));
    }

    @Override
    public void close() {}
}
