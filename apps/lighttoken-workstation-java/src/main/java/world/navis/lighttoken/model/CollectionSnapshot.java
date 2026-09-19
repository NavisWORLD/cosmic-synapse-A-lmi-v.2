package world.navis.lighttoken.model;

import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

public record CollectionSnapshot(Path source, List<TokenDetail> tokens, String collectionJson) {
    public CollectionSnapshot {
        source = Objects.requireNonNull(source, "source").toAbsolutePath().normalize();
        tokens = List.copyOf(tokens);
        Objects.requireNonNull(collectionJson, "collectionJson");
    }

    public TokenDetail token(String tokenId) {
        return tokens.stream()
                .filter(token -> token.summary().tokenId().equals(tokenId))
                .findFirst()
                .orElseThrow(() -> new IllegalArgumentException("unknown token id: " + tokenId));
    }

    public Map<String, TokenDetail> tokenMap() {
        Map<String, TokenDetail> result = new LinkedHashMap<>();
        for (TokenDetail token : tokens) {
            result.put(token.summary().tokenId(), token);
        }
        return Map.copyOf(result);
    }
}
