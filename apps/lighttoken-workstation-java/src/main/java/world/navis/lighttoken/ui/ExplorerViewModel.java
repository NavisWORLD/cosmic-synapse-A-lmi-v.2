package world.navis.lighttoken.ui;

import java.nio.file.Path;
import java.util.Objects;
import java.util.concurrent.CompletableFuture;
import world.navis.lighttoken.model.BackendInfo;
import world.navis.lighttoken.model.CollectionSnapshot;
import world.navis.lighttoken.model.QueryResult;
import world.navis.lighttoken.model.SearchFilters;
import world.navis.lighttoken.model.SimilarityMethod;
import world.navis.lighttoken.model.TokenDetail;
import world.navis.lighttoken.nativebridge.NativeEngine;
import world.navis.lighttoken.service.CollectionService;
import world.navis.lighttoken.service.SearchService;

public final class ExplorerViewModel implements AutoCloseable {
    private final NativeEngine engine;
    private final CollectionService collections;
    private final SearchService searches;

    private volatile CollectionSnapshot snapshot;
    private volatile TokenDetail selectedToken;
    private volatile QueryResult lastQueryResult;

    public ExplorerViewModel(
            NativeEngine engine, CollectionService collections, SearchService searches) {
        this.engine = Objects.requireNonNull(engine, "engine");
        this.collections = Objects.requireNonNull(collections, "collections");
        this.searches = Objects.requireNonNull(searches, "searches");
    }

    public CompletableFuture<CollectionSnapshot> openCollection(Path source) {
        return collections.open(source)
                .thenApply(
                        opened -> {
                            snapshot = opened;
                            selectedToken = opened.tokens().isEmpty() ? null : opened.tokens().getFirst();
                            lastQueryResult = null;
                            return opened;
                        });
    }

    public CollectionSnapshot snapshot() {
        return snapshot;
    }

    public TokenDetail selectedToken() {
        return selectedToken;
    }

    public QueryResult lastQueryResult() {
        return lastQueryResult;
    }

    public BackendInfo backendInfo() {
        return engine.backendInfo();
    }

    public TokenDetail selectToken(String tokenId) {
        CollectionSnapshot current = requireSnapshot();
        TokenDetail selected = current.token(Objects.requireNonNull(tokenId, "tokenId"));
        selectedToken = selected;
        return selected;
    }

    public CompletableFuture<QueryResult> search(
            String queryTokenId, SimilarityMethod method, int topK) {
        CollectionSnapshot current = requireSnapshot();
        if (topK <= 0) {
            throw new IllegalArgumentException("topK must be positive");
        }
        return searches.search(
                        current,
                        Objects.requireNonNull(queryTokenId, "queryTokenId"),
                        Objects.requireNonNull(method, "method"),
                        topK,
                        null,
                        SearchFilters.none())
                .thenApply(
                        result -> {
                            lastQueryResult = result;
                            return result;
                        });
    }

    private CollectionSnapshot requireSnapshot() {
        CollectionSnapshot current = snapshot;
        if (current == null) {
            throw new IllegalStateException("open a LightToken collection first");
        }
        return current;
    }

    @Override
    public void close() {}
}
