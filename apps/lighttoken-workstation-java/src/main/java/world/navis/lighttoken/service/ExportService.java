package world.navis.lighttoken.service;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Objects;
import java.util.concurrent.CompletableFuture;
import world.navis.lighttoken.model.QueryResult;
import world.navis.lighttoken.model.TokenDetail;

public final class ExportService implements AutoCloseable {
    private static final ObjectMapper JSON = new ObjectMapper();

    private final ServiceExecutor workers;

    public ExportService(ServiceExecutor workers) {
        this.workers = Objects.requireNonNull(workers, "workers");
    }

    public Path requireExplicitFile(Path destination) {
        if (destination == null || destination.toString().isBlank()) {
            throw new IllegalArgumentException("an explicit export file is required");
        }
        Path normalized = destination.toAbsolutePath().normalize();
        if (Files.isDirectory(normalized)) {
            throw new IllegalArgumentException("export destination must be a file: " + normalized);
        }
        if (normalized.getFileName() == null) {
            throw new IllegalArgumentException("export destination must include a file name");
        }
        return normalized;
    }

    public CompletableFuture<Path> exportToken(TokenDetail token, Path destination) {
        Objects.requireNonNull(token, "token");
        Path output = requireExplicitFile(destination);
        return workers.submit(
                () -> {
                    createParent(output);
                    Files.writeString(
                            output,
                            token.canonicalJson(),
                            StandardCharsets.UTF_8);
                    return output;
                });
    }

    public CompletableFuture<Path> exportQueryResults(QueryResult result, Path destination) {
        Objects.requireNonNull(result, "result");
        Path output = requireExplicitFile(destination);
        return workers.submit(
                () -> {
                    ObjectNode root = JSON.createObjectNode();
                    root.put("query_token_id", result.queryTokenId());
                    root.put("method", result.method().wireName());
                    root.put("backend", result.backend());
                    root.put("created_at", result.createdAt());
                    ArrayNode hits = root.putArray("hits");
                    for (var hit : result.hits()) {
                        ObjectNode node = hits.addObject();
                        node.put("token_id", hit.tokenId());
                        node.put("score", hit.score());
                        node.put("rank", hit.rank());
                    }
                    createParent(output);
                    Files.writeString(
                            output,
                            JSON.writerWithDefaultPrettyPrinter().writeValueAsString(root) + "\n",
                            StandardCharsets.UTF_8);
                    return output;
                });
    }

    private static void createParent(Path output) throws Exception {
        Path parent = output.getParent();
        if (parent != null) {
            Files.createDirectories(parent);
        }
    }

    @Override
    public void close() {}
}
