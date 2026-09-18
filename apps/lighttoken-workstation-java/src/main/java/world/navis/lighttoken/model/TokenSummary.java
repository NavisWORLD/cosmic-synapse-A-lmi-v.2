package world.navis.lighttoken.model;

public record TokenSummary(
        String tokenId,
        String timestamp,
        String sourceUri,
        String modality,
        String rawDataRef) {}
