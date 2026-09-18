package world.navis.lighttoken.model;

public record ValidationResult(
        boolean valid,
        String tokenId,
        int embeddingDimension,
        int spectralDimension) {}
