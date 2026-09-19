package world.navis.lighttoken.model;

import java.util.Objects;

public record WorkspaceTokenPayload(
        String tokenId,
        String sourcePath,
        String rawDataRef,
        String canonicalJson,
        boolean resolvable,
        boolean verified,
        boolean rawResolvable,
        String tokenSha256,
        String rawSha256) {
    public WorkspaceTokenPayload {
        Objects.requireNonNull(tokenId, "tokenId");
        Objects.requireNonNull(sourcePath, "sourcePath");
        Objects.requireNonNull(rawDataRef, "rawDataRef");
    }
}
