package world.navis.lighttoken.service;

import java.util.Objects;
import java.util.concurrent.CompletableFuture;
import world.navis.lighttoken.library.TokenRepository;
import world.navis.lighttoken.model.CollectionSnapshot;
import world.navis.lighttoken.model.QueryResult;
import world.navis.lighttoken.model.SearchFilters;
import world.navis.lighttoken.model.SearchRequest;
import world.navis.lighttoken.model.SimilarityMethod;
import world.navis.lighttoken.model.TokenDetail;
import world.navis.lighttoken.nativebridge.NativeEngine;

public final class SearchService implements AutoCloseable {
    private final NativeEngine engine;
    private final TokenRepository repository;
    private final ServiceExecutor workers;

    public SearchService(
            NativeEngine engine, TokenRepository repository, ServiceExecutor workers) {
        this.engine = Objects.requireNonNull(engine, "engine");
        this.repository = Objects.requireNonNull(repository, "repository");
        this.workers = Objects.requireNonNull(workers, "workers");
    }

    public CompletableFuture<QueryResult> search(
            CollectionSnapshot snapshot,
            String queryTokenId,
            SimilarityMethod method,
            int topK,
            Float threshold,
            SearchFilters filters) {
        Objects.requireNonNull(snapshot, "snapshot");
        Objects.requireNonNull(method, "method");
        SearchFilters safeFilters = filters == null ? SearchFilters.none() : filters;
        if (topK <= 0) {
            throw new IllegalArgumentException("topK must be positive");
        }
        return workers.submit(
                () -> {
                    TokenDetail query = snapshot.token(queryTokenId);
                    var nativeResult =
                            engine.searchJson(
                                    query.canonicalJson(),
                                    snapshot.collectionJson(),
                                    new SearchRequest(
                                            method,
                                            topK,
                                            threshold,
                                            safeFilters.modality(),
                                            safeFilters.sourcePrefix()));
                    String createdAt = query.summary().timestamp();
                    long queryId =
                            repository.recordQuery(
                                    queryTokenId, method.wireName(), nativeResult.backend(), createdAt);
                    for (var hit : nativeResult.hits()) {
                        repository.recordQueryHit(queryId, hit.rank(), hit.tokenId(), hit.score());
                    }
                    return new QueryResult(
                            queryTokenId,
                            method,
                            nativeResult.backend(),
                            nativeResult.hits(),
                            createdAt);
                });
    }

    @Override
    public void close() {}
}
