package world.navis.lighttoken.model;

import java.util.Objects;

public record SourceVerification(
        String sourceKind,
        String logicalSource,
        boolean resolvable,
        boolean verified,
        boolean rawResolvable,
        String tokenSha256,
        String rawSha256) {
    public SourceVerification {
        Objects.requireNonNull(sourceKind, "sourceKind");
        Objects.requireNonNull(logicalSource, "logicalSource");
    }

    public static SourceVerification unverified(String sourceKind, String logicalSource) {
        return new SourceVerification(
                sourceKind, logicalSource, true, false, false, null, null);
    }
}
