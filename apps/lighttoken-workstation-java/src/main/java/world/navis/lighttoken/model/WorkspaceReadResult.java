package world.navis.lighttoken.model;

import java.util.List;
import java.util.Objects;

public record WorkspaceReadResult(
        String sourceKind, boolean bundleVerified, List<WorkspaceTokenPayload> tokens) {
    public WorkspaceReadResult {
        Objects.requireNonNull(sourceKind, "sourceKind");
        tokens = List.copyOf(tokens);
    }
}
