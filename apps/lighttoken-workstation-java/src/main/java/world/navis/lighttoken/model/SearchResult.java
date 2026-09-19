package world.navis.lighttoken.model;

import java.util.List;

public record SearchResult(
        String queryTokenId,
        SimilarityMethod method,
        String backend,
        List<SearchHit> hits) {
    public SearchResult {
        hits = List.copyOf(hits);
    }
}
