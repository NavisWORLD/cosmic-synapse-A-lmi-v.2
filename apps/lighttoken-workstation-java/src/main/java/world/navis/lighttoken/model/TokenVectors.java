package world.navis.lighttoken.model;

import java.util.Arrays;

public record TokenVectors(String tokenId, float[] embedding, float[] spectralPower) {
    public TokenVectors {
        embedding = Arrays.copyOf(embedding, embedding.length);
        spectralPower = Arrays.copyOf(spectralPower, spectralPower.length);
    }

    @Override
    public float[] embedding() {
        return Arrays.copyOf(embedding, embedding.length);
    }

    @Override
    public float[] spectralPower() {
        return Arrays.copyOf(spectralPower, spectralPower.length);
    }
}
