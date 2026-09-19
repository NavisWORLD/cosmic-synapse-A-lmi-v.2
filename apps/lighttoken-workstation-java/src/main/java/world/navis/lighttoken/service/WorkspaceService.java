package world.navis.lighttoken.service;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.CompletableFuture;
import world.navis.lighttoken.library.TokenRepository;
import world.navis.lighttoken.model.CollectionSnapshot;
import world.navis.lighttoken.model.SourceVerification;
import world.navis.lighttoken.model.TokenDetail;
import world.navis.lighttoken.model.TokenSummary;
import world.navis.lighttoken.model.WorkspaceReadResult;
import world.navis.lighttoken.nativebridge.NativeEngine;

public final class WorkspaceService implements AutoCloseable {
    private static final ObjectMapper JSON = new ObjectMapper();

    private final CollectionService collections;
    private final NativeEngine engine;
    private final TokenRepository repository;
    private final ServiceExecutor workers;

    public WorkspaceService(CollectionService collections) {
        this.collections = Objects.requireNonNull(collections, "collections");
        this.engine = null;
        this.repository = null;
        this.workers = null;
    }

    public WorkspaceService(
            NativeEngine engine, TokenRepository repository, ServiceExecutor workers) {
        this.collections = null;
        this.engine = Objects.requireNonNull(engine, "engine");
        this.repository = Objects.requireNonNull(repository, "repository");
        this.workers = Objects.requireNonNull(workers, "workers");
    }

    public CompletableFuture<CollectionSnapshot> openWorkspace(Path path) {
        Path source = Objects.requireNonNull(path, "path").toAbsolutePath().normalize();
        if (engine == null) {
            return collections.open(source);
        }
        return workers.submit(() -> toSnapshot(engine.readWorkspace(source), source));
    }

    public CompletableFuture<CollectionSnapshot> openCosmos(Path path) {
        Path source = Objects.requireNonNull(path, "path").toAbsolutePath().normalize();
        if (engine == null) {
            return CompletableFuture.failedFuture(
                    new UnsupportedOperationException(
                            "verified .cosmos reads require the native workspace engine"));
        }
        return workers.submit(() -> toSnapshot(engine.readCosmos(source), source));
    }

    private CollectionSnapshot toSnapshot(WorkspaceReadResult read, Path source) throws Exception {
        List<TokenDetail> tokens = new ArrayList<>();
        List<String> collectionJson = new ArrayList<>();
        for (var payload : read.tokens()) {
            if (!payload.resolvable() || payload.canonicalJson() == null) {
                continue;
            }
            String canonical = payload.canonicalJson();
            JsonNode node = JSON.readTree(canonical);
            TokenSummary summary =
                    new TokenSummary(
                            payload.tokenId(),
                            requiredText(node, "timestamp"),
                            requiredText(node, "source_uri"),
                            requiredText(node, "modality"),
                            requiredText(node, "raw_data_ref"));
            long sourceId =
                    repository.upsertSource(
                            source,
                            read.sourceKind(),
                            payload.tokenSha256(),
                            summary.timestamp());
            repository.upsertToken(summary, sourceId, 1);
            SourceVerification verification =
                    new SourceVerification(
                            read.sourceKind(),
                            payload.sourcePath(),
                            payload.resolvable(),
                            payload.verified(),
                            payload.rawResolvable(),
                            payload.tokenSha256(),
                            payload.rawSha256());
            tokens.add(
                    new TokenDetail(
                            summary,
                            canonical.endsWith("\n") ? canonical : canonical + "\n",
                            engine.vectorsJson(canonical),
                            source,
                            verification));
            collectionJson.add(canonical.trim());
        }
        return new CollectionSnapshot(source, tokens, "[" + String.join(",", collectionJson) + "]");
    }

    private static String requiredText(JsonNode node, String field) {
        String value = node.path(field).asText("");
        if (value.isBlank()) {
            throw new IllegalArgumentException("LightToken is missing " + field);
        }
        return value;
    }

    @Override
    public void close() {}
}
