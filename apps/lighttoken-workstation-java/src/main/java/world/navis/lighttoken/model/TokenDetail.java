package world.navis.lighttoken.model;

import java.nio.file.Path;
import java.util.Objects;

public record TokenDetail(
        TokenSummary summary,
        String canonicalJson,
        TokenVectors vectors,
        Path sourcePath,
        SourceVerification verification) {
    public TokenDetail {
        Objects.requireNonNull(summary, "summary");
        Objects.requireNonNull(canonicalJson, "canonicalJson");
        Objects.requireNonNull(vectors, "vectors");
        Objects.requireNonNull(sourcePath, "sourcePath");
        Objects.requireNonNull(verification, "verification");
        sourcePath = sourcePath.toAbsolutePath().normalize();
    }

    public TokenDetail(
            TokenSummary summary,
            String canonicalJson,
            TokenVectors vectors,
            Path sourcePath) {
        this(
                summary,
                canonicalJson,
                vectors,
                sourcePath,
                SourceVerification.unverified(
                        "collection", sourcePath.toAbsolutePath().normalize().toString()));
    }
}
