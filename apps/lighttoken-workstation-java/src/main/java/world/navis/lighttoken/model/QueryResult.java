package world.navis.lighttoken.model;

import java.util.List;
import java.util.Objects;

public record QueryResult(
        String queryTokenId,
        SimilarityMethod method,
        String backend,
        List<SearchHit> hits,
        String createdAt) {
    public QueryResult {
        Objects.requireNonNull(queryTokenId, "queryTokenId");
        Objects.requireNonNull(method, "method");
        Objects.requireNonNull(backend, "backend");
        hits = List.copyOf(hits);
        Objects.requireNonNull(createdAt, "createdAt");
    }
}
