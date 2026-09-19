package world.navis.lighttoken.model;

public record SearchRequest(
        SimilarityMethod method,
        Integer topK,
        Float threshold,
        String modality,
        String sourcePrefix) {
    public SearchRequest {
        if (method == null) {
            throw new IllegalArgumentException("method is required");
        }
        if (topK != null && topK <= 0) {
            throw new IllegalArgumentException("topK must be positive when supplied");
        }
        if (threshold != null && !Float.isFinite(threshold)) {
            throw new IllegalArgumentException("threshold must be finite when supplied");
        }
    }
}
