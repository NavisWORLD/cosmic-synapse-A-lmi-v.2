package world.navis.lighttoken.model;

import java.nio.file.Path;
import java.util.Objects;

public record TokenDetail(
        TokenSummary summary,
        String canonicalJson,
        TokenVectors vectors,
        Path sourcePath) {
    public TokenDetail {
        Objects.requireNonNull(summary, "summary");
        Objects.requireNonNull(canonicalJson, "canonicalJson");
        Objects.requireNonNull(vectors, "vectors");
        Objects.requireNonNull(sourcePath, "sourcePath");
        sourcePath = sourcePath.toAbsolutePath().normalize();
    }
}
