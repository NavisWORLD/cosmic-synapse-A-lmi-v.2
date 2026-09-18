package world.navis.lighttoken.model;

public record ComparisonResult(
        String leftTokenId,
        String rightTokenId,
        SimilarityMethod method,
        float score,
        String backend) {}
