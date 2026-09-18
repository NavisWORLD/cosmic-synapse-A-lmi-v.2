package world.navis.lighttoken.model;

public record QueryRecord(
        long queryId,
        String queryTokenId,
        String method,
        String backend,
        String createdAt,
        int hitCount) {}
