package world.navis.lighttoken.service;

import java.nio.file.Path;
import java.util.Objects;
import java.util.concurrent.CompletableFuture;
import world.navis.lighttoken.model.CollectionSnapshot;

public final class WorkspaceService implements AutoCloseable {
    private final CollectionService collections;

    public WorkspaceService(CollectionService collections) {
        this.collections = Objects.requireNonNull(collections, "collections");
    }

    public CompletableFuture<CollectionSnapshot> openWorkspace(Path path) {
        return collections.open(path);
    }

    public CompletableFuture<CollectionSnapshot> openCosmos(Path path) {
        return CompletableFuture.failedFuture(
                new UnsupportedOperationException(
                        "verified .cosmos reads are enabled by the Task 11 integrity boundary"));
    }

    @Override
    public void close() {}
}
